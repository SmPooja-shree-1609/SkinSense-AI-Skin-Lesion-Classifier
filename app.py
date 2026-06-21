"""
app.py
------
SkinSense — Final frontend.
Polished, modern, medical-AI style Streamlit UI.
All predictions come exclusively from predictor.py → model.predict().
No logic changes to the verified backend.
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError
from predictor import load_model, load_label_encoder, predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinSense — Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

# ── Global CSS — soft medical-AI palette, clean cards ─────────────────────────
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background-color: #f4f7fb;
    }

    /* Hero section card */
    .hero-card {
        background: linear-gradient(135deg, #1a6b8a 0%, #2196a6 60%, #3dbdbd 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(26, 107, 138, 0.25);
    }
    .hero-card h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-card .subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 1rem;
    }
    .hero-card .description {
        font-size: 0.92rem;
        opacity: 0.82;
        line-height: 1.6;
    }

    /* Upload card */
    .upload-card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }
    .upload-card h3 {
        color: #1a6b8a;
        margin-top: 0;
        font-size: 1.1rem;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }

    /* Top prediction highlight box */
    .top-prediction {
        background: linear-gradient(135deg, #e8f8f5 0%, #d4f1f4 100%);
        border-left: 5px solid #1a6b8a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .top-prediction .class-code {
        font-size: 0.85rem;
        color: #1a6b8a;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .top-prediction .class-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0d3d52;
        margin: 0.2rem 0;
    }
    .top-prediction .confidence {
        font-size: 1rem;
        color: #2196a6;
        font-weight: 600;
    }

    /* Probability row */
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.55rem;
        gap: 0.6rem;
    }
    .prob-label {
        min-width: 220px;
        font-size: 0.88rem;
        color: #333;
    }
    .prob-label .code {
        font-weight: 700;
        color: #1a6b8a;
    }
    .prob-bar-bg {
        flex: 1;
        background: #e8eef3;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 10px;
        border-radius: 6px;
        background: linear-gradient(90deg, #1a6b8a, #3dbdbd);
    }
    .prob-bar-fill.top {
        background: linear-gradient(90deg, #1a6b8a, #2196a6);
    }
    .prob-pct {
        min-width: 48px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: #444;
    }

    /* Disclaimer card */
    .disclaimer-card {
        background: #fff8e7;
        border: 1px solid #f0d080;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #6b5a1e;
        line-height: 1.6;
    }
    .disclaimer-card strong {
        color: #4a3c0a;
    }

    /* Section label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #1a6b8a;
        margin-bottom: 0.6rem;
    }

    /* Hide Streamlit default top padding */
    .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Readable class name mapping ────────────────────────────────────────────────
# Maps the short training codes to human-readable medical names.
# You can update these labels here without touching predictor.py.
CLASS_LABELS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

def get_readable_name(code: str) -> str:
    """Return the human-readable label for a class code, or the code itself."""
    return CLASS_LABELS.get(code, code)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-card">
    <h1>🔬 SkinSense</h1>
    <div class="subtitle">Intelligent Skin Lesion Classification using Deep Learning</div>
    <div class="description">
        SkinSense uses a trained EfficientNet model to classify dermoscopic skin lesion images
        into one of 7 clinical categories from the HAM10000 dataset.
        Upload an image below to receive an AI-assisted classification.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL LOADING (silent, cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()

@st.cache_resource(show_spinner=False)
def get_encoder():
    return load_label_encoder()

# Load model — halt with a clear error if the file is missing
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load label encoder — optional
label_encoder = get_encoder()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2b — HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:white; border-radius:12px; padding:1.1rem 1.5rem;
            box-shadow:0 2px 10px rgba(0,0,0,0.06); margin-bottom:1.5rem;
            display:flex; gap:2rem; flex-wrap:wrap;">
    <div style="flex:1; min-width:140px; text-align:center;">
        <div style="font-size:1.6rem;">📤</div>
        <div style="font-weight:700; color:#1a6b8a; font-size:0.9rem;">1. Upload</div>
        <div style="font-size:0.82rem; color:#555;">Select a dermoscopic lesion image (JPG or PNG)</div>
    </div>
    <div style="flex:1; min-width:140px; text-align:center;">
        <div style="font-size:1.6rem;">🔍</div>
        <div style="font-weight:700; color:#1a6b8a; font-size:0.9rem;">2. Analyse</div>
        <div style="font-size:0.82rem; color:#555;">Click <em>Analyse Image</em> to run the model</div>
    </div>
    <div style="flex:1; min-width:140px; text-align:center;">
        <div style="font-size:1.6rem;">📊</div>
        <div style="font-weight:700; color:#1a6b8a; font-size:0.9rem;">3. Review</div>
        <div style="font-size:0.82rem; color:#555;">Check the predicted class and all probabilities</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Step 1 — Upload Image</div>', unsafe_allow_html=True)

with st.container():
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    st.caption("Supported formats: JPG, JPEG, PNG  ·  For best results, use a clear dermoscopic image.")

# ── Image preview and prediction ───────────────────────────────────────────────
if uploaded_file is not None:

    # Validate and open the image
    try:
        image = Image.open(uploaded_file)
        image.load()  # Force decode — catches corrupt files early
    except UnidentifiedImageError:
        st.error("❌ The uploaded file could not be read as an image. Please upload a valid JPG or PNG.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Image error: {e}")
        st.stop()

    # Show preview
    st.markdown('<div class="section-label" style="margin-top:1rem;">Image Preview</div>', unsafe_allow_html=True)
    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(image, use_container_width=True)
    with col_info:
        st.markdown(f"""
        **File:** {uploaded_file.name}
        
        **Size:** {image.size[0]} × {image.size[1]} px
        
        **Mode:** {image.mode}
        
        *Image will be resized to 224×224 for the model.*
        """)

    st.markdown("---")

    # ── Predict button ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 2 — Run Classification</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍  Analyse Image", type="primary", use_container_width=True)

    if predict_btn:

        # Run prediction — all logic lives in predictor.py, unchanged
        try:
            with st.spinner("Analysing image with the trained model..."):
                result = predict(image, model, label_encoder)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        top_code = result["label"]
        top_name = get_readable_name(top_code)
        confidence = result["confidence"]

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4 — RESULTS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Results</div>', unsafe_allow_html=True)

        # Top prediction highlight
        st.markdown(f"""
        <div class="top-prediction">
            <div class="class-code">Top Prediction · {top_code}</div>
            <div class="class-name">{top_name}</div>
            <div class="confidence">Confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

        # All 7 class probabilities — sorted descending
        st.markdown("**All Class Probabilities**")

        sorted_scores = sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for code, score in sorted_scores:
            readable = get_readable_name(code)
            is_top = (code == top_code)
            bar_class = "prob-bar-fill top" if is_top else "prob-bar-fill"
            label_style = "font-weight:700;" if is_top else ""

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{label_style}">
                    <span class="code">{code}</span> — {readable}
                </div>
                <div class="prob-bar-bg">
                    <div class="{bar_class}" style="width:{score}%;"></div>
                </div>
                <div class="prob-pct">{score}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Debug block — collapsed, available for verification ────────────────
        with st.expander("🛠 Debug Info"):
            st.write(f"**Input shape:** `{result['image_shape']}` — expected `(1, 224, 224, 3)`")
            prob_sum = round(sum(result["raw_predictions"][0]), 6)
            st.write(f"**Probability sum:** `{prob_sum}` — should be ≈ 1.0")
            st.write(f"**Predicted index:** `{result['predicted_index']}`")
            st.write(f"**Predicted class:** `{result['label']}` — {get_readable_name(result['label'])}")
            st.write(f"**Label encoder:** {'loaded' if label_encoder is not None else 'not found — using fallback'}")
            st.write("**Raw softmax vector:**")
            st.write(result["raw_predictions"][0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Medical Disclaimer</strong><br>
    SkinSense is an educational and research tool built to demonstrate AI-assisted
    skin lesion classification. It is <strong>not a medical diagnostic device</strong>
    and must not be used as a substitute for professional clinical evaluation.
    All results should be interpreted by a qualified dermatologist or healthcare provider.
    Do not make any medical decisions based solely on this tool's output.
</div>
""", unsafe_allow_html=True)
