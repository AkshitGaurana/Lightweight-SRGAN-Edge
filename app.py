"""
app.py -- Streamlit demo for Lightweight SRGAN (8 RCB) vs Full SRGAN (16 RCB)
Research paper: "Single Image Super-Resolution Through Lightweight SRGAN for Edge Deployment"
"""

import io
import os
import sys
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from backend import bicubic_upscale, _load_weights, _load_image_as_tensor, _tensor_to_image, _ensure_dir, _DEFAULT_WEIGHTS_PATH, _UPSCALE_FACTOR
from metrics import evaluate_method, compute_psnr, compute_ssim
from model import SRResNet

import torch

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lightweight SRGAN | Edge SR Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS -- Dark premium theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1729 50%, #0d1a0f 100%);
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 41, 0.95);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99, 179, 237, 0.5); }

.method-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.tag-bicubic  { background: rgba(107,114,128,0.3); color: #9ca3af; border: 1px solid #4b5563; }
.tag-lite     { background: rgba(16,185,129,0.2);  color: #34d399; border: 1px solid #059669; }
.tag-full     { background: rgba(139,92,246,0.2);  color: #a78bfa; border: 1px solid #7c3aed; }

.psnr-val { font-size: 2rem; font-weight: 700; color: #63b3ed; }
.ssim-val { font-size: 2rem; font-weight: 700; color: #68d391; }
.time-val { font-size: 1.6rem; font-weight: 600; color: #f6ad55; }

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #68d391, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub {
    color: #718096;
    font-size: 0.95rem;
    margin-bottom: 24px;
}

/* Divider */
hr { border-color: rgba(99, 179, 237, 0.1) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2b6cb0, #285e61);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Image captions */
.img-caption {
    text-align: center;
    font-size: 0.8rem;
    color: #718096;
    margin-top: 4px;
}

/* Warning / info boxes */
.info-box {
    background: rgba(66, 153, 225, 0.1);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #bee3f8;
    margin: 12px 0;
}
.warn-box {
    background: rgba(237, 137, 54, 0.1);
    border-left: 3px solid #f6ad55;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #fbd38d;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: build SRResNet for a given num_rcb and try to load weights
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(num_rcb: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRResNet(in_channels=3, out_channels=3, channels=64,
                     num_rcb=num_rcb, upscale=4)
    loaded = _load_weights(model, _DEFAULT_WEIGHTS_PATH, device)
    model = model.to(device)
    model.eval()
    return model, device, loaded


# ---------------------------------------------------------------------------
# Helper: run SRGAN inference on an in-memory BGR numpy array
# ---------------------------------------------------------------------------
def srgan_infer(model, device, bgr_image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device, non_blocking=True)
    with torch.no_grad():
        out = model(tensor).clamp(0, 1)
    out_np = out.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Helper: bicubic on in-memory array
# ---------------------------------------------------------------------------
def bicubic_infer(bgr_image: np.ndarray) -> np.ndarray:
    h, w = bgr_image.shape[:2]
    return cv2.resize(bgr_image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Helper: numpy BGR -> PIL for display
# ---------------------------------------------------------------------------
def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Helper: PIL/numpy image to bytes for download
# ---------------------------------------------------------------------------
def img_to_bytes(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Any standard image format. Works best on low-resolution inputs.",
    )

    st.markdown("---")
    st.markdown("### Model Options")
    show_lite = st.checkbox("Lightweight SRGAN (8 RCBs)", value=True,
                            help="Generator with 8 residual blocks — faster, edge-ready")
    show_full = st.checkbox("Full SRGAN (16 RCBs)", value=True,
                            help="Original generator with 16 residual blocks")
    show_bicubic = st.checkbox("Bicubic Baseline", value=True,
                               help="Classical OpenCV bicubic 4× interpolation")

    st.markdown("---")
    st.markdown("### Benchmark & Processing")
    run_metrics = st.checkbox("Compute PSNR / SSIM", value=True,
                              help="Uses the input as HR ground truth, downscales 4x, then SR back up")
    optimize_text = st.checkbox("Optimize for Text / Documents", value=True,
                                help="Reduces GAN hallucination artifacts around letters and flat backgrounds")

    st.markdown("---")
    st.markdown(
        """
        <div style='color:#4a5568;font-size:0.78rem;'>
        📄 <b>Research Paper</b><br>
        Single Image Super-Resolution Through Lightweight SRGAN for Edge Deployment<br><br>
        SRM Institute of Science and Technology · 2025
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# MAIN HEADER
# ===========================================================================
st.markdown('<div class="hero-title">🔬 Lightweight SRGAN — Edge SR Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">4× Image Super-Resolution · 8-Block vs 16-Block Generator Comparison · Bicubic Baseline</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ===========================================================================
# Load models (cached)
# ===========================================================================
with st.spinner("Loading models..."):
    model_lite, device_lite, lite_loaded = get_model(8)
    model_full, device_full, full_loaded = get_model(16)

weights_available = lite_loaded or full_loaded
if not weights_available:
    st.markdown(
        '<div class="warn-box">⚠️  Pretrained SRGAN weights not found at '
        '<code>results/pretrained_models/SRGAN_x4-ImageNet.pth.tar</code>. '
        'SRGAN outputs will use <b>random weights</b> (distorted). '
        'The <b>Bicubic baseline</b> always produces clean results. '
        'Download real weights to unlock full SRGAN quality.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="info-box">✅ Pretrained weights loaded successfully. '
        f'Running on <b>{device_lite}</b>.</div>',
        unsafe_allow_html=True,
    )

device_label = str(device_lite).upper()
st.markdown(
    f'<div class="info-box">🖥️  Inference device: <b>{device_label}</b></div>',
    unsafe_allow_html=True,
)

# ===========================================================================
# Image input
# ===========================================================================
if uploaded is None:
    # Default sample image
    sample_path = os.path.join(_DIR, "figure", "comic.png")
    if os.path.isfile(sample_path):
        hr_bgr = cv2.imread(sample_path)
        st.markdown(
            '<div class="info-box">📷 Using sample image <b>figure/comic.png</b>. '
            'Upload your own image in the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No sample image found. Please upload an image using the sidebar.")
        st.stop()
else:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    hr_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if hr_bgr is None:
        st.error("Could not decode the uploaded image. Please try another file.")
        st.stop()

h, w = hr_bgr.shape[:2]
lh, lw = h // 4, w // 4
lr_bgr = cv2.resize(hr_bgr, (lw, lh), interpolation=cv2.INTER_CUBIC)

st.markdown(f"**Input:** `{w}×{h}` px  →  LR simulated at `{lw}×{lh}` px  →  SR target: `{w*4 if run_metrics else w}×{h*4 if run_metrics else h}` px")

# ===========================================================================
# Run inference
# ===========================================================================
results = {}   # method -> {"image": bgr_ndarray, "time_ms": float}
metrics_data = []

with st.spinner("Running super-resolution..."):

    if show_bicubic:
        t0 = time.perf_counter()
        bic_out = bicubic_infer(lr_bgr)
        bic_ms = (time.perf_counter() - t0) * 1000
        results["Bicubic"] = {"image": bic_out, "time_ms": round(bic_ms, 1)}

    if show_lite:
        t0 = time.perf_counter()
        lite_out = srgan_infer(model_lite, device_lite, lr_bgr)
        if optimize_text:
            bic_tmp = bicubic_infer(lr_bgr)
            lite_out = cv2.addWeighted(lite_out, 0.65, bic_tmp, 0.35, 0)
            lite_out = cv2.bilateralFilter(lite_out, d=5, sigmaColor=35, sigmaSpace=35)
        lite_ms = (time.perf_counter() - t0) * 1000
        results["SRGAN-Lite\n(8 RCBs)"] = {"image": lite_out, "time_ms": round(lite_ms, 1)}

    if show_full:
        t0 = time.perf_counter()
        full_out = srgan_infer(model_full, device_full, lr_bgr)
        if optimize_text:
            bic_tmp = bicubic_infer(lr_bgr)
            full_out = cv2.addWeighted(full_out, 0.65, bic_tmp, 0.35, 0)
            full_out = cv2.bilateralFilter(full_out, d=5, sigmaColor=35, sigmaSpace=35)
        full_ms = (time.perf_counter() - t0) * 1000
        results["SRGAN-Full\n(16 RCBs)"] = {"image": full_out, "time_ms": round(full_ms, 1)}

    if run_metrics and results:
        # Crop HR to same size as SR output for fair comparison
        for label, data in results.items():
            sr = data["image"]
            sh, sw = sr.shape[:2]
            ch, cw = min(h * 4, sh), min(w * 4, sw)
            hr_rs = cv2.resize(hr_bgr, (sw, sh), interpolation=cv2.INTER_CUBIC)
            hr_crop = hr_rs[:ch, :cw]
            sr_crop = sr[:ch, :cw]
            psnr = compute_psnr(hr_crop, sr_crop)
            ssim = compute_ssim(hr_crop, sr_crop)
            metrics_data.append({
                "Method":        label.replace("\n", " "),
                "PSNR (dB)":     round(psnr, 2),
                "SSIM":          round(ssim, 4),
                "Inference (ms)": data["time_ms"],
            })

# ===========================================================================
# Display images
# ===========================================================================
st.markdown("## 📸 Results")

n_cols = 1 + len(results)   # LR + each method
cols = st.columns(n_cols)

with cols[0]:
    st.image(bgr_to_pil(lr_bgr), use_container_width=True)
    st.markdown('<div class="img-caption">🔽 LR Input (4× downsampled)</div>', unsafe_allow_html=True)

tag_map = {
    "Bicubic":             ("tag-bicubic",  "Bicubic"),
    "SRGAN-Lite\n(8 RCBs)": ("tag-lite",   "SRGAN-Lite · 8 RCBs"),
    "SRGAN-Full\n(16 RCBs)": ("tag-full",  "SRGAN-Full · 16 RCBs"),
}

for idx, (label, data) in enumerate(results.items(), start=1):
    with cols[idx]:
        status = ""
        if "SRGAN" in label:
            loaded = lite_loaded if "Lite" in label else full_loaded
            status = " ✅" if loaded else " ⚠️ rand weights"
        st.image(bgr_to_pil(data["image"]), use_container_width=True)
        tag_cls, tag_txt = tag_map.get(label, ("tag-bicubic", label))
        st.markdown(
            f'<div class="img-caption">'
            f'<span class="method-tag {tag_cls}">{tag_txt}{status}</span><br>'
            f'⏱ {data["time_ms"]} ms'
            f'</div>',
            unsafe_allow_html=True,
        )

# ===========================================================================
# Metrics table
# ===========================================================================
if run_metrics and metrics_data:
    st.markdown("---")
    st.markdown("## 📊 Quality Metrics")
    st.markdown(
        '<div class="info-box">Computed using standard SR benchmark protocol: '
        'input image treated as HR ground truth → downscaled 4× (bicubic) → '
        'upscaled by each method → PSNR/SSIM vs original.</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(len(metrics_data))
    tag_cls_map = {"Bicubic": "tag-bicubic", "SRGAN-Lite (8 RCBs)": "tag-lite", "SRGAN-Full (16 RCBs)": "tag-full"}

    for col, row in zip(metric_cols, metrics_data):
        tag_cls = tag_cls_map.get(row["Method"], "tag-bicubic")
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <span class="method-tag {tag_cls}">{row['Method']}</span>
                  <div style="margin-top:8px;">
                    <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;letter-spacing:1px;">PSNR</div>
                    <div class="psnr-val">{row['PSNR (dB)']} <span style="font-size:1rem;color:#4a5568;">dB</span></div>
                  </div>
                  <div style="margin-top:10px;">
                    <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;letter-spacing:1px;">SSIM</div>
                    <div class="ssim-val">{row['SSIM']}</div>
                  </div>
                  <div style="margin-top:10px;">
                    <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;letter-spacing:1px;">Inference</div>
                    <div class="time-val">{row['Inference (ms)']} <span style="font-size:1rem;color:#4a5568;">ms</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show as table too
    import pandas as pd
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Store metrics in session state for DOCX patching
    st.session_state["metrics"] = metrics_data

# ===========================================================================
# Downloads
# ===========================================================================
st.markdown("---")
st.markdown("## 💾 Download Results")
dl_cols = st.columns(len(results))
for col, (label, data) in zip(dl_cols, results.items()):
    safe_name = label.replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "")
    col.download_button(
        label=f"⬇ {label.replace(chr(10), ' ')}",
        data=img_to_bytes(data["image"]),
        file_name=f"sr_{safe_name}.jpg",
        mime="image/jpeg",
    )

# ===========================================================================
# Architecture info panel
# ===========================================================================
st.markdown("---")
with st.expander("ℹ️ About this demo", expanded=False):
    st.markdown("""
    ### Lightweight SRGAN for Edge Deployment

    This demo implements the proposed architecture from the research paper:
    *"Single Image Super-Resolution Through Lightweight SRGAN for Edge Deployment"*

    | Component | Original SRGAN | Proposed (Ours) |
    |---|---|---|
    | Residual Blocks | 16 | **8** |
    | Channels | 64 | 64 |
    | Upscale Factor | 4× | 4× |
    | Discriminator | VGG-style | VGG-style (unchanged) |
    | Loss | VGG + Adversarial | VGG + Adversarial (unchanged) |
    | Approx. FLOPs | 2× | **1×** (halved) |

    **Benchmark methodology**: Input image is used as ground-truth HR. It is
    downsampled 4× (bicubic) to simulate a real LR capture, then each SR method
    reconstructs the HR image. PSNR and SSIM are computed against the original.
    This matches the standard Set5/Set14/BSD100 evaluation protocol.

    **Why PSNR may look similar across methods**: Without pretrained GAN weights,
    both SRGAN variants fall back to structural upsampling without learned textures.
    With real weights, SRGAN produces perceptually richer outputs (higher MOS)
    while PSNR values are often comparable to or slightly below bicubic — consistent
    with published SRGAN literature (Ledig et al., 2017).
    """)
