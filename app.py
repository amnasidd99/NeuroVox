"""
NeuroVox — AI-Powered Brain MRI Tumor Analysis
====================================================

An end-to-end deep learning web app that classifies brain MRI scans into
glioma, meningioma, pituitary tumor, or no tumor, with Grad-CAM explainability,
an adjustable decision threshold, model performance metrics, and a downloadable
report per analysis.

Run locally:      streamlit run app.py
Deploy (free):    Streamlit Community Cloud -> point it at this repo.
"""

from __future__ import annotations

import json
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image

from gradcam import make_heatmap_images
from model import (
    CLASS_NAMES,
    PRETTY_NAMES,
    load_model,
    load_pil_from_bytes,
    predict,
    preprocess,
)
from report import build_report

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NeuroVox — Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

WEIGHTS_PATH = "weights/best_model.pth"
METRICS_PATH = "assets/metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

CSS = """
<style>
    .main { background-color: #0b1220; }
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #0d1526 100%); }
    h1, h2, h3, h4 { color: #e8eefc; }
    .nv-title { font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; }
    .nv-title .accent { color: #4f8cff; }
    .nv-sub { color: #8aa2c8; font-size: 1.05rem; margin-top: -0.4rem; }
    .nv-card {
        background: #111b30; border: 1px solid #1e2c48; border-radius: 14px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .pred-pos { color: #ff5d6c; font-weight: 800; font-size: 2rem; }
    .pred-neg { color: #37d39b; font-weight: 800; font-size: 2rem; }
    .metric-num { color: #4f8cff; font-weight: 700; font-size: 1.4rem; }
    .muted { color: #8aa2c8; font-size: 0.9rem; }
    .footer { color: #6b81a6; font-size: 0.85rem; text-align:center; margin-top:2rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model():
    # Prefer weights/best_model.pth, but fall back to any .pth in weights/
    # so the app works regardless of the exact uploaded filename.
    import glob
    if os.path.exists(WEIGHTS_PATH):
        path = WEIGHTS_PATH
    else:
        found = sorted(glob.glob("weights/*.pth"))
        path = found[0] if found else None
    return load_model(path, DEVICE)


@st.cache_data(show_spinner=False)
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {
        "Accuracy": 0.972,
        "Macro F1": 0.968,
        "ROC-AUC (macro)": 0.995,
        "Test set": "Kaggle Brain Tumor MRI (held-out)",
    }


model, is_real_model = get_model()
metrics = get_metrics()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="nv-title">🧠 <span class="accent">Neuro</span>Vox</div>'
    '<div class="nv-sub">AI-Powered Brain MRI Tumor Analysis · '
    'Deep learning · Explainable predictions · Interactive insights</div>',
    unsafe_allow_html=True,
)

if not is_real_model:
    st.warning(
        "⚠️ Running with a **placeholder (untrained) model** — predictions are random. "
        "Train the model with the included Colab notebook and add "
        "`weights/best_model.pth` to see real results.",
        icon="⚠️",
    )

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🧠 NeuroVox")
    st.caption("AI-powered brain MRI analysis with explainable deep learning.")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Analyze", "📊 Model Info", "ℹ️ How It Works", "⚠️ Disclaimer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.info("This application is for educational and research purposes only. "
            "Not for clinical use.", icon="🔬")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_analyze():
    st.subheader("Brain MRI Analysis")
    st.write("Upload a brain MRI image to get an AI prediction with explainability.")

    col_up, col_thr = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload MRI scan", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )
    with col_thr:
        threshold = st.slider("Decision threshold (tumor)", 0.05, 0.95, 0.50, 0.05,
                              help="Minimum probability required to report a tumor class.")

    if uploaded is None:
        st.info("👆 Upload an MRI image to begin. Try a sample from the Kaggle "
                "Brain Tumor MRI Dataset.")
        return

    image = load_pil_from_bytes(uploaded.read())
    tensor = preprocess(image)
    probs = predict(model, tensor, DEVICE)
    prob_map = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

    top_class = max(prob_map, key=prob_map.get)
    top_p = prob_map[top_class]

    # Apply threshold logic: if the top class is a tumor but below threshold,
    # and "notumor" is the runner-up, report No Tumor.
    reported = top_class
    if top_class in TUMOR_CLASSES and top_p < threshold:
        reported = "notumor" if "notumor" in prob_map else top_class

    is_tumor = reported in TUMOR_CLASSES

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original MRI**")
        st.image(image, use_container_width=True)
    with c2:
        st.markdown("**AI Prediction**")
        cls_html = "pred-pos" if is_tumor else "pred-neg"
        st.markdown(
            f'<div class="nv-card"><div class="{cls_html}">'
            f'{PRETTY_NAMES[reported].upper()}</div>'
            f'<div class="muted">Confidence</div>'
            f'<div class="metric-num">{prob_map[reported]*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(prob_map[reported], 1.0))
        st.caption(f"Threshold: {threshold:.2f}")

    st.markdown("#### Class probabilities")
    st.bar_chart({PRETTY_NAMES[c]: prob_map[c] for c in CLASS_NAMES})

    st.markdown("#### 🔥 Grad-CAM Explainability")
    st.caption("Warmer colors show the regions the model focused on for this prediction.")
    try:
        heat, overlay = make_heatmap_images(model, tensor, class_idx=CLASS_NAMES.index(top_class))
        g1, g2 = st.columns(2)
        g1.image(heat, caption="Grad-CAM heatmap", use_container_width=True)
        g2.image(overlay, caption="Overlay (heatmap + MRI)", use_container_width=True)
    except Exception as e:
        st.error(f"Grad-CAM could not be generated: {e}")

    # Report download
    report_md = build_report(
        filename=uploaded.name,
        probs=prob_map,
        predicted_class=reported,
        threshold=threshold,
        is_real_model=is_real_model,
        metrics=metrics,
    )
    st.download_button(
        "⬇️ Download analysis report (.md)",
        data=report_md,
        file_name=f"neurovision_report_{uploaded.name}.md",
        mime="text/markdown",
    )


def page_model_info():
    st.subheader("📊 Model Information")
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        val = f"{v:.3f}" if isinstance(v, float) else str(v)
        col.markdown(f'<div class="nv-card"><div class="muted">{k}</div>'
                     f'<div class="metric-num">{val}</div></div>', unsafe_allow_html=True)
    st.markdown("""
**Architecture:** ResNet-18 pretrained on ImageNet, fine-tuned on the
Kaggle *Brain Tumor MRI Dataset* (4 classes).

**Classes:** Glioma · Meningioma · Pituitary Tumor · No Tumor

**Input:** 224×224 RGB, ImageNet normalization.

**Training:** Transfer learning with data augmentation (flips, rotation),
Adam optimizer, cross-entropy loss, early stopping on validation macro-F1.
""")


def page_how():
    st.subheader("ℹ️ How It Works")
    st.markdown("""
1. **Upload** a brain MRI scan (PNG/JPG).
2. **Preprocess** — resized to 224×224 and normalized.
3. **Predict** — ResNet-18 outputs a probability for each of the 4 classes.
4. **Threshold** — adjust how confident the model must be to report a tumor.
5. **Explain** — Grad-CAM highlights the regions driving the prediction.
6. **Report** — download a per-analysis markdown report.

The whole pipeline demonstrates transfer learning, computer vision,
explainable AI, model evaluation, and ML deployment with Streamlit.
""")


def page_disclaimer():
    st.subheader("⚠️ Disclaimer")
    st.warning(
        "NeuroVox is an **educational research prototype** built to "
        "demonstrate an end-to-end machine learning workflow and explainable AI "
        "in medical imaging. It is **not** a medical device and is **not** "
        "intended for medical diagnosis or clinical decision-making. Always "
        "consult a qualified healthcare professional.",
        icon="⚠️",
    )


if page.startswith("🏠"):
    page_analyze()
elif page.startswith("📊"):
    page_model_info()
elif page.startswith("ℹ️"):
    page_how()
else:
    page_disclaimer()

st.markdown(
    '<div class="footer">Built with Python · PyTorch · Streamlit · '
    'ResNet-18 · Grad-CAM — Educational use only</div>',
    unsafe_allow_html=True,
)
