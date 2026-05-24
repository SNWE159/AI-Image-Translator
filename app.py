"""
AI Image Language Translator
Streamlit Community Cloud — Production Build
─────────────────────────────────────────────
Stack : EasyOCR · deep-translator · OpenCV-headless · Pillow · Streamlit
Author: AI Image Translator Project
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from deep_translator import GoogleTranslator
import io
import os
import time
import urllib.request

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ─ must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Image Language Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AI Image Language Translator — Detect, Translate & Replace text in images.",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:          #0a0e1a;
    --surface:     #111827;
    --surface2:    #1a2235;
    --accent:      #00e5ff;
    --accent2:     #7c3aed;
    --text:        #e2e8f0;
    --text-muted:  #64748b;
    --success:     #10b981;
    --border:      #1e293b;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: var(--bg);
    color: var(--text);
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a0e1a 100%);
}

/* ── Hero ── */
.hero-header { text-align: center; padding: 2.5rem 0 1.2rem; }
.hero-title {
    font-size: clamp(1.8rem, 5vw, 3rem);
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: .4rem;
}
.hero-subtitle {
    font-size: .95rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: .05em;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin: .2rem;
}
.badge-cyan   { background:rgba(0,229,255,.12);  color:var(--accent);  border:1px solid rgba(0,229,255,.3); }
.badge-purple { background:rgba(124,58,237,.12); color:#a78bfa;        border:1px solid rgba(124,58,237,.3); }
.badge-green  { background:rgba(16,185,129,.12); color:var(--success); border:1px solid rgba(16,185,129,.3); }

/* ── Metrics ── */
.metric-row { display:flex; gap:1rem; margin:1rem 0; flex-wrap:wrap; }
.metric-box {
    flex:1; min-width:110px;
    background:var(--surface2);
    border:1px solid var(--border);
    border-radius:10px;
    padding:1rem;
    text-align:center;
}
.metric-value { font-size:1.7rem; font-weight:700; color:var(--accent); font-family:'JetBrains Mono',monospace; }
.metric-label { font-size:.72rem; color:var(--text-muted); margin-top:.2rem; }

/* ── Pipeline steps ── */
.steps { display:flex; gap:.5rem; align-items:center; margin:.8rem 0; flex-wrap:wrap; }
.step  { display:flex; align-items:center; gap:.4rem; font-size:.78rem; }
.step-num {
    width:24px; height:24px; border-radius:50%;
    background:var(--surface2); border:1px solid var(--border);
    display:flex; align-items:center; justify-content:center;
    font-size:.7rem; font-weight:600; color:var(--text-muted);
    font-family:'JetBrains Mono',monospace;
}
.step-num.active { background:rgba(0,229,255,.2);   border-color:var(--accent);  color:var(--accent);  }
.step-num.done   { background:rgba(16,185,129,.2);  border-color:var(--success); color:var(--success); }
.step-arrow { color:var(--border); }

/* ── Detection table ── */
.det-table { width:100%; border-collapse:collapse; font-size:.8rem; }
.det-table th { color:var(--text-muted); font-weight:600; padding:.5rem .75rem; text-align:left; border-bottom:1px solid var(--border); }
.det-table td { padding:.5rem .75rem; border-bottom:1px solid #0f172a; color:var(--text); }
.det-table tr:last-child td { border-bottom:none; }
.conf-bar-wrap { background:#0f172a; border-radius:4px; height:6px; width:80px; }
.conf-bar { background:linear-gradient(90deg,var(--accent),var(--accent2)); height:6px; border-radius:4px; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .6rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    transition: all .2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.45) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #047857) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .6rem 2rem !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
}
div[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid var(--border); }
.stProgress > div > div { background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; }
div[data-testid="stAlert"] { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MAX_IMAGE_DIM = 1200   # Streamlit Cloud RAM is limited; keep this ≤1200

LANG_NAMES = {
    "es": "Spanish",    "fr": "French",     "de": "German",
    "it": "Italian",    "pt": "Portuguese", "nl": "Dutch",
    "ja": "Japanese",   "ko": "Korean",     "ar": "Arabic",
}

# EasyOCR language codes to load (keep list small to save memory on free tier)
OCR_LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "ar"]

# ══════════════════════════════════════════════════════════════════════════════
# FONT  ─ download DejaVu Bold once into /tmp (writable on Streamlit Cloud)
# ══════════════════════════════════════════════════════════════════════════════
FONT_URL  = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
FONT_PATH = "/tmp/DejaVuSans-Bold.ttf"

def ensure_font() -> str:
    """Return path to a bold TTF; download to /tmp if not present."""
    # 1. Try /tmp (persists across reruns in same session)
    if os.path.exists(FONT_PATH):
        return FONT_PATH
    # 2. Try common Linux system paths (may exist on the Cloud image)
    system_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in system_paths:
        if os.path.exists(p):
            return p
    # 3. Download from GitHub (one-time, ~170 KB)
    try:
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        return FONT_PATH
    except Exception:
        return ""   # Will fall back to PIL default


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ocr_reader() -> easyocr.Reader:
    """
    Load EasyOCR once and keep it in memory for all sessions.
    gpu=False  → required on Streamlit Community Cloud (CPU only).
    verbose=False → suppresses console noise.
    """
    return easyocr.Reader(OCR_LANGS, gpu=False, verbose=False)


@st.cache_data(show_spinner=False)
def cached_font_path() -> str:
    return ensure_font()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def resize_if_large(pil_img: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════
def run_ocr(reader: easyocr.Reader, pil_img: Image.Image) -> list:
    """
    Run EasyOCR on a PIL image.
    Returns list of dicts: {text, bbox:(x1,y1,x2,y2), confidence}
    """
    cv_img   = pil_to_cv(pil_img)
    raw      = reader.readtext(cv_img, detail=1, paragraph=False)
    results  = []
    for (bbox_pts, text, conf) in raw:
        if conf < 0.15 or not text.strip():
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        results.append({
            "text":       text.strip(),
            "bbox":       (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            "confidence": conf,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════
def translate_text(text: str) -> str:
    """
    Translate text to English using deep-translator's GoogleTranslator.
    source='auto'  → language auto-detection (no API key needed).
    Falls back to original text on any error.
    """
    try:
        if not text.strip():
            return text
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result if result else text
    except Exception:
        return text


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REMOVAL (OpenCV inpainting)
# ══════════════════════════════════════════════════════════════════════════════
def remove_text_inpaint(cv_img: np.ndarray, detections: list) -> np.ndarray:
    """
    Build a binary mask over all bounding boxes and use INPAINT_TELEA
    to reconstruct the background behind each text region.
    """
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
    h, w = cv_img.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        pad = 5
        mask[max(0, y1-pad):min(h, y2+pad),
             max(0, x1-pad):min(w, x2+pad)] = 255
    return cv2.inpaint(cv_img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
# FONT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load DejaVu Bold at `size`; fall back to PIL default."""
    fp = cached_font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_text_in_box(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int):
    """Binary-search the largest font size where `text` fits in (box_w × box_h)."""
    for size in range(min(box_h, 48), 5, -1):
        font = get_font(size)
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:                     # Pillow < 9.2
            tw, th = draw.textsize(text, font=font)
        if tw <= box_w and th <= box_h:
            return font, tw, th
    return get_font(6), box_w, box_h


# ══════════════════════════════════════════════════════════════════════════════
# TEXT RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def render_translated_text(pil_img: Image.Image, detections: list) -> Image.Image:
    """
    Draw translated text back into every bounding box:
      1. White fill rectangle (for legibility)
      2. Black text, centered & auto-sized
    """
    img  = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        translated = det.get("translated", "").strip()
        if not translated:
            continue

        x1, y1, x2, y2 = det["bbox"]
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)

        font, tw, th = fit_text_in_box(draw, translated, box_w, box_h)

        # White background fill
        pad = 2
        draw.rectangle(
            [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
            fill=(255, 255, 255),
        )
        # Centered text
        tx = x1 + (box_w - tw) // 2
        ty = y1 + (box_h - th) // 2
        draw.text((tx, ty), translated, fill=(15, 15, 15), font=font)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def process_image(pil_img: Image.Image,
                  reader: easyocr.Reader,
                  progress_cb=None):
    """
    Orchestrate the full OCR → translate → inpaint → render pipeline.
    Returns (result_pil_image, detections_list).
    """
    def step(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    step("🔬 Preprocessing image…", 5)
    pil_img = resize_if_large(pil_img)

    step("🔍 Detecting text (EasyOCR)…", 20)
    detections = run_ocr(reader, pil_img)

    if not detections:
        step("⚠️ No text detected in image.", 100)
        return pil_img, []

    step(f"🌐 Translating {len(detections)} text region(s)…", 48)
    for det in detections:
        det["translated"] = translate_text(det["text"])

    step("🖼️ Removing original text (inpainting)…", 70)
    cv_img        = pil_to_cv(pil_img)
    inpainted_cv  = remove_text_inpaint(cv_img, detections)
    inpainted_pil = cv_to_pil(inpainted_cv)

    step("✏️ Writing translated text onto image…", 88)
    result = render_translated_text(inpainted_pil, detections)

    step("✅ Done!", 100)
    return result, detections


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def detections_html(detections: list) -> str:
    rows = ""
    for i, det in enumerate(detections, 1):
        conf  = int(det["confidence"] * 100)
        orig  = det["text"].replace("<", "&lt;").replace(">", "&gt;")
        trans = det.get("translated", "—").replace("<", "&lt;").replace(">", "&gt;")
        same  = orig.lower() == trans.lower()
        trans_cell = f'<span style="color:var(--text-muted);font-style:italic;">{trans} (unchanged)</span>' if same else trans
        rows += f"""
        <tr>
          <td><span class="badge badge-cyan">{i}</span></td>
          <td>{orig}</td>
          <td>{trans_cell}</td>
          <td>
            <div style="display:flex;align-items:center;gap:6px;">
              <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{conf}%"></div>
              </div>
              <span style="font-size:.72rem;color:#64748b;font-family:'JetBrains Mono',monospace">{conf}%</span>
            </div>
          </td>
        </tr>"""
    return f"""
    <table class="det-table">
      <thead><tr>
        <th>#</th>
        <th>Original Text</th>
        <th>Translated → English</th>
        <th>Confidence</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def img_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():

    # ── Hero header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🌐 AI Image Language Translator</div>
        <div class="hero-subtitle">
            Detect · Translate · Replace &nbsp;—&nbsp; No training · No paid APIs
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Language badges ──────────────────────────────────────────────────────
    badges = " ".join(
        f'<span class="badge badge-purple">{name}</span>'
        for name in LANG_NAMES.values()
    )
    st.markdown(
        f'<div style="text-align:center;margin-bottom:1.5rem;">{badges}</div>',
        unsafe_allow_html=True,
    )

    # ── Load OCR model ───────────────────────────────────────────────────────
    with st.spinner("⏳ Loading OCR model — this takes ~40 s on first boot…"):
        reader = load_ocr_reader()

    # ── Two-column layout ────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: upload + controls ──────────────────────────────────────────────
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload Image")

        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_orig = Image.open(uploaded).convert("RGB")
            st.image(pil_orig, caption="Original Image", use_container_width=True)
            w, h = pil_orig.size
            size_kb = round(len(uploaded.getvalue()) / 1024, 1)
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box">
                <div class="metric-value">{w}</div>
                <div class="metric-label">Width px</div>
              </div>
              <div class="metric-box">
                <div class="metric-value">{h}</div>
                <div class="metric-label">Height px</div>
              </div>
              <div class="metric-box">
                <div class="metric-value">{size_kb}</div>
                <div class="metric-label">Size KB</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Translate button ─────────────────────────────────────────────────
        if uploaded:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="steps">
              <div class="step">
                <div class="step-num active">1</div><span>OCR</span>
              </div>
              <div class="step-arrow">→</div>
              <div class="step">
                <div class="step-num active">2</div><span>Translate</span>
              </div>
              <div class="step-arrow">→</div>
              <div class="step">
                <div class="step-num active">3</div><span>Inpaint</span>
              </div>
              <div class="step-arrow">→</div>
              <div class="step">
                <div class="step-num active">4</div><span>Render</span>
              </div>
            </div>""", unsafe_allow_html=True)

            run_clicked = st.button("🚀 Translate Image", key="run_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            # Store trigger in session_state so Streamlit reruns work correctly
            if run_clicked:
                st.session_state["trigger"] = True

    # ── RIGHT: placeholder or result ─────────────────────────────────────────
    if not uploaded:
        with right:
            st.markdown("""
            <div class="card" style="min-height:320px;display:flex;
                 align-items:center;justify-content:center;
                 flex-direction:column;gap:.6rem;">
                <div style="font-size:3rem;">🖼️</div>
                <div style="color:var(--text-muted);font-size:.9rem;">
                    Translated image will appear here
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Processing ───────────────────────────────────────────────────────────
    if uploaded and st.session_state.get("trigger"):

        # Reset trigger so re-uploading a new file doesn't auto-process
        st.session_state["trigger"] = False

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_cb(msg: str, pct: int):
            status_text.markdown(
                f'<p style="color:var(--accent);font-size:.85rem;'
                f'font-family:\'JetBrains Mono\',monospace;margin:0;">{msg}</p>',
                unsafe_allow_html=True,
            )
            progress_bar.progress(pct)

        try:
            pil_orig = Image.open(uploaded).convert("RGB")
            result_img, detections = process_image(pil_orig, reader, progress_cb)
            time.sleep(0.25)
            progress_bar.empty()
            status_text.empty()

            # ── Result image ─────────────────────────────────────────────────
            with right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### ✅ Translated Image")
                st.image(result_img, caption="Translated → English", use_container_width=True)

                result_bytes = img_to_bytes(result_img)
                st.download_button(
                    label="⬇️ Download Translated Image",
                    data=result_bytes,
                    file_name="translated_image.png",
                    mime="image/png",
                    key="dl_btn",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Stats + detection table ───────────────────────────────────────
            if detections:
                n_translated = sum(
                    1 for d in detections
                    if d.get("translated") and
                       d["translated"].lower() != d["text"].lower()
                )
                st.markdown("---")
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box">
                    <div class="metric-value">{len(detections)}</div>
                    <div class="metric-label">Regions Found</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-value">{n_translated}</div>
                    <div class="metric-label">Translated</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-value">{len(detections) - n_translated}</div>
                    <div class="metric-label">Already English</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### 📋 Detection & Translation Details")
                st.markdown(
                    f'<div class="card" style="overflow-x:auto;">'
                    f'{detections_html(detections)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("ℹ️ No text regions were detected in this image. "
                        "Try a sharper image with higher-contrast text.")

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Processing error: {exc}")
            with st.expander("Show traceback"):
                st.exception(exc)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 .8rem;
         color:#1e293b;font-size:.72rem;letter-spacing:.04em;">
        Powered by &nbsp;EasyOCR · deep-translator · OpenCV · Pillow · Streamlit
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
