# 🌐 AI Image Language Translator

> Detect multilingual text in images, translate it to English, and replace it — live in Streamlit Community Cloud.

---

## 📁 Project Structure

```
your-repo/
├── app.py                   ← Main Streamlit application
├── requirements.txt         ← Python dependencies
├── packages.txt             ← System (apt) dependencies  ← CRITICAL for Cloud
├── .streamlit/
│   └── config.toml          ← Streamlit theme + server config
└── README.md
```

---

## 🚀 Deploy to Streamlit Community Cloud (Step-by-Step)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — AI Image Translator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step 2 — Create Streamlit Cloud App

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

> ⏳ First deploy takes **5–10 minutes** — PyTorch + EasyOCR models are large.
> Subsequent cold starts take ~40 seconds (model loading).

---

## ⚙️ Why Each File Matters on Streamlit Cloud

| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages installed via pip |
| `packages.txt` | **Critical** — installs `libgl1`, `fonts-dejavu-core`, etc. via apt before pip. Without this, OpenCV and font rendering fail silently. |
| `.streamlit/config.toml` | Sets dark theme, disables CORS issues, enables fast reruns |

---

## 🔧 Cloud-Specific Optimisations in `app.py`

| Change | Why |
|--------|-----|
| `gpu=False` in EasyOCR | Streamlit Cloud has no GPU |
| `MAX_IMAGE_DIM = 1200` | Free tier has ~800 MB RAM limit |
| Font downloaded to `/tmp/` | `/tmp` is writable on Cloud; home dir may not be |
| `packages.txt` includes `fonts-dejavu-core` | System font path `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` is used first (no download needed) |
| `opencv-python-headless` | Headless build avoids Qt/GTK GUI libs that aren't available on Cloud |
| `@st.cache_resource` for OCR reader | Model loaded once per server instance, not per user session |
| Session state `trigger` flag | Prevents double-processing on Streamlit reruns |

---

## 🌍 Supported Languages

| Language   | Detection |
|------------|-----------|
| Spanish    | ✅ |
| French     | ✅ |
| German     | ✅ |
| Italian    | ✅ |
| Portuguese | ✅ |
| Dutch      | ✅ |
| Japanese   | ✅ |
| Korean     | ✅ |
| Arabic     | ✅ |
| Chinese (Simplified) | ✅ |
| English    | ✅ (passthrough) |

> **Sinhala** requires a separate EasyOCR community model not bundled by default.
> Add `'si'` to `OCR_LANGS` in `app.py` if you install it manually.

---

## 🏗️ Pipeline

```
Upload Image
     ↓
Resize to ≤ 1200px (saves Cloud RAM)
     ↓
EasyOCR  →  text + bounding boxes + confidence
     ↓
deep-translator GoogleTranslator(source="auto", target="en")
     ↓
cv2.inpaint(INPAINT_TELEA)  →  remove original text
     ↓
PIL ImageDraw  →  render translated text (auto font size)
     ↓
Streamlit display + PNG download
```

---

## 🐛 Troubleshooting

**"ModuleNotFoundError: cv2"**
→ Check `packages.txt` has `libgl1` and `libglib2.0-0`. These must exist.

**"OSError: cannot open resource" (font)**
→ Ensure `packages.txt` has `fonts-dejavu-core`. The app also auto-downloads the font to `/tmp` as a fallback.

**App crashes with MemoryError**
→ Image too large. Lower `MAX_IMAGE_DIM` to `800` in `app.py`.

**Slow first load**
→ Normal — EasyOCR downloads ~200 MB of model weights on first boot.
→ Free tier cold starts add ~30–60 s on top of this.

**Translation fails / returns original text**
→ `deep-translator` calls Google's free endpoint. If the app is deployed in a region where Google is blocked, try replacing with `MyMemoryTranslator` from the same library.

---

## 📦 Tech Stack

| Library | Role |
|---------|------|
| Streamlit | Web UI |
| EasyOCR | OCR + text detection |
| deep-translator | Free translation wrapper |
| opencv-python-headless | Inpainting + image ops |
| Pillow | Text rendering |
| NumPy | Array operations |
| PyTorch (CPU) | EasyOCR neural net backend |

---

## 📄 License

MIT — free to use, modify, and deploy.
