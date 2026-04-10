"""
PDF Graphics Comparator — Flask/Render version
High-accuracy page-by-page and graphics-level PDF comparison.
No OpenCV required. Uses PyMuPDF + scikit-image + imagehash.
"""

import io
import os
import uuid
import zipfile
import json
import threading
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import fitz          # PyMuPDF
import imagehash
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageEnhance
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity as ssim
from flask import (Flask, render_template, request, jsonify,
                   send_file, session, redirect, url_for)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "pdfcmp-secret-2024")

# ── In-memory job store (resets on redeploy — files are temporary anyway) ──────
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

UPLOAD_FOLDER = "/tmp/pdfcmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PageResult:
    page_no: int
    similarity_score: float
    changed_pixels_pct: float
    status: str
    notes: str

@dataclass
class ImageMatchResult:
    page_no_source: int
    img_index_source: int
    page_no_target: int
    img_index_target: int
    hash_distance: int
    status: str

# ══════════════════════════════════════════════════════════════════════════════
#  PDF HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_pdf_pages(pdf_bytes: bytes, zoom: float = 2.0) -> List[Image.Image]:
    """Render each PDF page as a PIL Image at given zoom."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    matrix = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    doc.close()
    return pages

def extract_pdf_images(pdf_bytes: bytes) -> List[Dict]:
    """Extract all embedded raster images from a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # Skip tiny icons/decorations (< 32×32)
                if pil_img.width < 32 or pil_img.height < 32:
                    continue
                extracted.append({
                    "page_no": page_index + 1,
                    "img_index": img_index + 1,
                    "ext": base_image.get("ext", "png"),
                    "image": pil_img,
                    "hash": imagehash.phash(pil_img, hash_size=16),  # larger hash = more accurate
                    "size": (pil_img.width, pil_img.height),
                })
            except Exception:
                continue
    doc.close()
    return extracted

def extract_pdf_text_blocks(pdf_bytes: bytes) -> List[Dict]:
    """Extract text blocks per page for text-level diff."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        blocks.append({"page_no": page_index + 1, "text": text})
    doc.close()
    return blocks

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE COMPARISON — HIGH ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
def resize_to_common(img1: Image.Image, img2: Image.Image) -> Tuple[Image.Image, Image.Image]:
    w = min(img1.width, img2.width)
    h = min(img1.height, img2.height)
    return img1.resize((w, h), Image.LANCZOS), img2.resize((w, h), Image.LANCZOS)

def preprocess_for_comparison(img: Image.Image) -> Image.Image:
    """Normalise contrast and sharpness before comparison to reduce false positives."""
    img = ImageEnhance.Contrast(img).enhance(1.1)
    return img

def compare_images_ssim(img1: Image.Image, img2: Image.Image,
                        diff_threshold: int = 30,
                        min_region_area: int = 80) -> Dict:
    """
    Multi-pass high-accuracy SSIM comparison.
    Returns ssim_score, changed_pct, diff_mask, marked images.
    """
    img1, img2 = resize_to_common(img1, img2)
    img1 = preprocess_for_comparison(img1)
    img2 = preprocess_for_comparison(img2)

    # Pass 1: Greyscale SSIM
    a = np.array(img1.convert("L"), dtype=np.float32)
    b = np.array(img2.convert("L"), dtype=np.float32)
    score_grey, diff_grey = ssim(a, b, full=True, data_range=255.0)

    # Pass 2: Per-channel SSIM for colour accuracy
    scores_ch = []
    diffs_ch  = []
    for ch in range(3):
        ac = np.array(img1)[..., ch].astype(np.float32)
        bc = np.array(img2)[..., ch].astype(np.float32)
        s, d = ssim(ac, bc, full=True, data_range=255.0)
        scores_ch.append(s)
        diffs_ch.append(d)

    score_colour = float(np.mean(scores_ch))
    diff_colour  = np.mean(diffs_ch, axis=0)

    # Combined score: weighted average (colour more sensitive)
    score = 0.35 * score_grey + 0.65 * score_colour

    # Build difference mask from colour diff
    diff_inverted = 1.0 - diff_colour
    # Normalise to 0–255 range
    diff_norm = (diff_inverted * 255).clip(0, 255).astype(np.uint8)
    thresh_arr = (diff_norm > diff_threshold).astype(np.uint8)

    # Morphological closing: fill small gaps in detected regions
    from scipy.ndimage import binary_closing, binary_dilation
    closed = binary_closing(thresh_arr, iterations=2)
    dilated = binary_dilation(closed, iterations=1)
    labeled = label(dilated.astype(np.uint8))

    marked_1 = img1.copy().convert("RGBA")
    marked_2 = img2.copy().convert("RGBA")
    draw1 = ImageDraw.Draw(marked_1)
    draw2 = ImageDraw.Draw(marked_2)

    overlay1 = Image.new("RGBA", img1.size, (0, 0, 0, 0))
    overlay2 = Image.new("RGBA", img2.size, (0, 0, 0, 0))
    ov_draw1 = ImageDraw.Draw(overlay1)
    ov_draw2 = ImageDraw.Draw(overlay2)

    changed_area = 0
    regions_found = 0

    for region in regionprops(labeled):
        if region.area < min_region_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        padding = 4
        min_col = max(0, min_col - padding)
        min_row = max(0, min_row - padding)
        max_col = min(img1.width, max_col + padding)
        max_row = min(img1.height, max_row + padding)

        changed_area += (max_col - min_col) * (max_row - min_row)
        regions_found += 1

        # Red border + semi-transparent fill
        draw1.rectangle([min_col, min_row, max_col, max_row], outline=(220, 30, 30, 255), width=3)
        draw2.rectangle([min_col, min_row, max_col, max_row], outline=(220, 30, 30, 255), width=3)
        ov_draw1.rectangle([min_col, min_row, max_col, max_row], fill=(255, 60, 60, 55))
        ov_draw2.rectangle([min_col, min_row, max_col, max_row], fill=(255, 60, 60, 55))

    marked_1 = Image.alpha_composite(marked_1, overlay1).convert("RGB")
    marked_2 = Image.alpha_composite(marked_2, overlay2).convert("RGB")

    total_area = img1.width * img1.height
    changed_pct = (changed_area / total_area) * 100 if total_area else 0.0

    # Diff mask: amplify for visibility
    diff_vis = Image.fromarray((diff_norm * 3).clip(0, 255).astype(np.uint8), mode="L")
    diff_vis = diff_vis.convert("RGB")

    return {
        "ssim_score":    float(score),
        "ssim_grey":     float(score_grey),
        "ssim_colour":   float(score_colour),
        "changed_pct":   float(changed_pct),
        "regions_found": regions_found,
        "diff_mask":     diff_vis,
        "marked_source": marked_1,
        "marked_target": marked_2,
    }

def compare_extracted_images(source_imgs: List[Dict],
                              target_imgs: List[Dict],
                              max_hash_distance: int = 10) -> List[ImageMatchResult]:
    """Match embedded images by perceptual hash, greedy best-match."""
    results = []
    used_target = set()

    for s in source_imgs:
        best_match = None
        best_dist  = 9999
        best_idx   = None
        for t_idx, t in enumerate(target_imgs):
            if t_idx in used_target:
                continue
            dist = s["hash"] - t["hash"]
            if dist < best_dist:
                best_dist  = dist
                best_match = t
                best_idx   = t_idx

        if best_match is not None:
            used_target.add(best_idx)
            status = "Matched" if best_dist <= max_hash_distance else "Different"
        else:
            best_match = {"page_no": -1, "img_index": -1}
            best_dist  = 999
            status     = "No Match Found"

        results.append(ImageMatchResult(
            page_no_source    = s["page_no"],
            img_index_source  = s["img_index"],
            page_no_target    = best_match["page_no"],
            img_index_target  = best_match["img_index"],
            hash_distance     = best_dist,
            status            = status,
        ))

    # Source images with no match in target
    all_source_pages = {(s["page_no"], s["img_index"]) for s in source_imgs}
    matched_sources  = {(r.page_no_source, r.img_index_source) for r in results}
    for s in source_imgs:
        key = (s["page_no"], s["img_index"])
        if key not in matched_sources:
            results.append(ImageMatchResult(s["page_no"], s["img_index"], -1, -1, 999, "Missing in Target"))

    return results

# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT COMPARISON ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════
def compare_documents(source_pages, target_pages,
                      page_threshold=0.985) -> Tuple[List[PageResult], Dict]:
    max_pages    = max(len(source_pages), len(target_pages))
    page_results = []
    evidence     = {}

    for i in range(max_pages):
        if i >= len(source_pages):
            page_results.append(PageResult(i+1, 0.0, 100.0, "Missing in Source",
                                           "Page exists only in target document"))
            continue
        if i >= len(target_pages):
            page_results.append(PageResult(i+1, 0.0, 100.0, "Missing in Target",
                                           "Page exists only in source document"))
            continue

        cmp = compare_images_ssim(source_pages[i], target_pages[i])
        sim     = cmp["ssim_score"]
        changed = cmp["changed_pct"]

        if sim >= page_threshold:
            status = "Match"
            notes  = "No material visual difference detected"
        elif sim >= 0.96:
            status = "Minor Difference"
            notes  = f"Small variation detected — {cmp['regions_found']} region(s) flagged"
        elif sim >= 0.90:
            status = "Moderate Difference"
            notes  = f"Noticeable difference — {cmp['regions_found']} region(s) flagged"
        else:
            status = "Major Difference"
            notes  = f"Significant page-level difference — {cmp['regions_found']} region(s) flagged"

        page_results.append(PageResult(
            page_no            = i + 1,
            similarity_score   = round(sim, 4),
            changed_pixels_pct = round(changed, 2),
            status             = status,
            notes              = notes,
        ))
        evidence[i + 1] = cmp

    return page_results, evidence

# ══════════════════════════════════════════════════════════════════════════════
#  ZIP REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def build_zip_report(page_results, image_results, evidence) -> bytes:
    import csv
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Page summary CSV
        page_rows = [asdict(r) for r in page_results]
        page_csv  = io.StringIO()
        if page_rows:
            w = csv.DictWriter(page_csv, fieldnames=page_rows[0].keys())
            w.writeheader(); w.writerows(page_rows)
        zf.writestr("page_comparison.csv", page_csv.getvalue())

        # Graphics summary CSV
        img_rows = [asdict(r) for r in image_results]
        img_csv  = io.StringIO()
        if img_rows:
            w = csv.DictWriter(img_csv, fieldnames=img_rows[0].keys())
            w.writeheader(); w.writerows(img_rows)
        zf.writestr("graphics_comparison.csv", img_csv.getvalue())

        # Evidence images
        for page_no, ev in evidence.items():
            if ev.get("marked_source"):
                zf.writestr(f"evidence/page_{page_no:03d}_source_marked.png",
                            img_to_png_bytes(ev["marked_source"]))
            if ev.get("marked_target"):
                zf.writestr(f"evidence/page_{page_no:03d}_target_marked.png",
                            img_to_png_bytes(ev["marked_target"]))
            if ev.get("diff_mask"):
                zf.writestr(f"evidence/page_{page_no:03d}_diff_mask.png",
                            img_to_png_bytes(ev["diff_mask"]))

    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND JOB RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_comparison_job(job_id: str, source_bytes: bytes, target_bytes: bytes,
                       zoom: float, page_threshold: float, hash_threshold: int):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "rendering"
            _jobs[job_id]["progress"] = 5

        source_pages = render_pdf_pages(source_bytes, zoom=zoom)
        target_pages = render_pdf_pages(target_bytes, zoom=zoom)

        with _jobs_lock:
            _jobs[job_id]["status"] = "extracting_graphics"
            _jobs[job_id]["progress"] = 20

        source_imgs = extract_pdf_images(source_bytes)
        target_imgs = extract_pdf_images(target_bytes)

        with _jobs_lock:
            _jobs[job_id]["status"] = "comparing_pages"
            _jobs[job_id]["progress"] = 35

        page_results, evidence = compare_documents(
            source_pages, target_pages, page_threshold=page_threshold)

        with _jobs_lock:
            _jobs[job_id]["status"] = "comparing_graphics"
            _jobs[job_id]["progress"] = 80

        image_results = compare_extracted_images(
            source_imgs, target_imgs, max_hash_distance=hash_threshold)

        with _jobs_lock:
            _jobs[job_id]["status"] = "building_report"
            _jobs[job_id]["progress"] = 90

        zip_bytes = build_zip_report(page_results, image_results, evidence)

        # Serialise page results for JSON
        page_summary = [asdict(r) for r in page_results]
        img_summary  = [asdict(r) for r in image_results]

        # Encode evidence images as base64 for frontend display
        import base64
        evidence_b64 = {}
        for page_no, ev in evidence.items():
            evidence_b64[page_no] = {}
            for key in ("marked_source", "marked_target", "diff_mask"):
                if ev.get(key):
                    evidence_b64[page_no][key] = base64.b64encode(
                        img_to_png_bytes(ev[key])).decode()

        # Stats
        total  = len(page_results)
        match  = sum(1 for r in page_results if r.status == "Match")
        minor  = sum(1 for r in page_results if r.status == "Minor Difference")
        mod    = sum(1 for r in page_results if r.status == "Moderate Difference")
        major  = sum(1 for r in page_results if r.status == "Major Difference")
        miss   = sum(1 for r in page_results if "Missing" in r.status)

        img_matched = sum(1 for r in image_results if r.status == "Matched")
        img_diff    = sum(1 for r in image_results if r.status == "Different")
        img_miss    = sum(1 for r in image_results if "Missing" in r.status or "No Match" in r.status)

        with _jobs_lock:
            _jobs[job_id].update({
                "status":        "done",
                "progress":      100,
                "page_results":  page_summary,
                "img_results":   img_summary,
                "evidence":      evidence_b64,
                "zip_bytes":     zip_bytes,
                "stats": {
                    "total": total, "match": match,
                    "minor": minor, "moderate": mod,
                    "major": major, "missing": miss,
                    "img_total":   len(image_results),
                    "img_matched": img_matched,
                    "img_diff":    img_diff,
                    "img_missing": img_miss,
                },
            })

    except Exception as e:
        import traceback
        with _jobs_lock:
            _jobs[job_id]["status"]  = "error"
            _jobs[job_id]["error"]   = str(e)
            _jobs[job_id]["traceback"] = traceback.format_exc()

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    source_file = request.files.get("source_pdf")
    target_file = request.files.get("target_pdf")
    if not source_file or not target_file:
        return jsonify({"error": "Both PDF files are required."}), 400

    zoom            = float(request.form.get("zoom", 2.0))
    page_threshold  = float(request.form.get("page_threshold", 0.985))
    hash_threshold  = int(request.form.get("hash_threshold", 10))

    # Cap zoom to avoid OOM on Render free tier
    zoom = min(zoom, 3.0)

    source_bytes = source_file.read()
    target_bytes = target_file.read()

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status":   "queued",
            "progress": 0,
            "source_name": source_file.filename,
            "target_name": target_file.filename,
        }

    t = threading.Thread(
        target=run_comparison_job,
        args=(job_id, source_bytes, target_bytes, zoom, page_threshold, hash_threshold),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    # Don't send zip_bytes or evidence over status endpoint
    safe = {k: v for k, v in job.items() if k not in ("zip_bytes", "evidence")}
    return jsonify(safe)

@app.route("/results/<job_id>")
def results(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Results not ready"}), 404
    return jsonify({
        "stats":        job["stats"],
        "page_results": job["page_results"],
        "img_results":  job["img_results"],
        "evidence":     job["evidence"],
    })

@app.route("/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        return "Report not ready", 404
    return send_file(
        io.BytesIO(job["zip_bytes"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name="pdf_comparison_report.zip",
    )

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
