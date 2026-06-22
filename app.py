"""
Cura - Medical document intelligence web app.

Upload a prescription or lab report image and get a clean, patient-friendly
summary. All document understanding (classification, OCR, NER, abnormality
detection, Gemini summarization) happens in the existing Phase 3 pipeline --
this file is presentation only.
"""
import json
import os
import re
import sys
import tempfile
import traceback
from html import escape
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif"]

st.set_page_config(
    page_title="Cura",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CUSTOM CSS -- single block, edit here for any visual changes
# ============================================================================
FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""
st.markdown(FONT_LINKS, unsafe_allow_html=True)

# NOTE: this block must start with "<style" as the very first characters of the
# string. CommonMark treats <script>/<pre>/<style> as a special HTML block that
# is immune to blank lines -- but only if the block *starts* with that tag. If
# anything else (like the <link> tags above) precedes <style> in the same
# st.markdown call, the block type changes and the first blank line inside the
# CSS terminates raw-HTML parsing early, dumping the rest of the CSS as visible
# page text.
CUSTOM_CSS = """
<style>
/* palette: #FFFDFB, #FBE7F1, #D7C6FF, #8FA9FF, #2D3A5E -- weighted toward
   purple/periwinkle (lilac + primary), not pink: pink (#FBE7F1) is kept out
   of large surfaces (backgrounds/cards) and only used sparingly so the app
   reads as a calm purple medical theme, not a pink one. */
:root {
    --cura-bg-top: #FFFDFB;
    --cura-bg-bottom: #F5F0FD;
    --cura-text: #2D3A5E;
    --cura-text-soft: #5B6680;
    --cura-muted: #8B93AC;
    --cura-primary: #8FA9FF;
    --cura-primary-dark: #2D3A5E;
    --cura-primary-soft: #BCCBFF;
    --cura-lilac: #D7C6FF;
    --cura-lavender: #F0E9FF;
    --cura-card-bg: #FFFFFF;
    --cura-border: #E1D6F7;
    --cura-dark: #2D3A5E;
    --cura-pink-accent: #FBE7F1;
}

/* font, applied via inheritance from html/body plus form controls (which
   don't inherit by default). Deliberately NOT using [data-testid] or
   [class*="st-"] here -- those are broad enough to also match Streamlit's
   icon elements (e.g. the expander chevron), and overriding font-family on
   those breaks the icon glyph and shows its raw ligature name as text. */
html, body, .stMarkdown, button, input, textarea {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, var(--cura-bg-top) 0%, var(--cura-bg-bottom) 100%);
}

/* Streamlit fades every individual output element to opacity:0.33 (with a
   1s transition) while it's "stale" during a script rerun -- confirmed by
   reading Streamlit's own bundled frontend source (static/js/index*.js):
   each [data-testid="stElementContainer"] gets a data-stale flag, and a
   styled-component spreads {opacity:.33, transition:"opacity 1s ease-in
   0.5s"} onto it when stale. Since every element on the page has this
   wrapper, this is what reads as "the whole page blurring" during the
   pipeline call. Overriding it directly here, plus the ancestor containers
   as a harmless extra safety net. */
[data-testid="stElementContainer"],
html, body, #root,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"],
.main, .block-container {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
html, body, p, span, div, label, li {
    color: var(--cura-text);
}

/* ---------------- navbar ---------------- */
.cura-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
    border-bottom: 1px solid var(--cura-border);
}
.cura-navbar-logo {
    font-family: 'Pacifico', cursive;
    font-size: 3.2rem;
    font-weight: 400;
    color: var(--cura-primary-dark);
    line-height: 1;
}
.cura-navbar-tag {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cura-muted);
    font-weight: 700;
}

/* ---------------- hero ---------------- */
.cura-hero {
    text-align: center;
    padding: 0.6rem 1rem 2.2rem;
}
.cura-hero-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: var(--cura-pink-accent);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--cura-primary-dark);
    margin-bottom: 1.1rem;
}
.cura-hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    line-height: 1.18;
    color: var(--cura-text);
    max-width: 760px;
    margin: 0 auto;
}
.cura-hero-accent {
    background: linear-gradient(90deg, var(--cura-primary), var(--cura-lilac));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.cura-hero-sub {
    font-size: 1.05rem;
    color: var(--cura-text-soft);
    margin-top: 0.9rem;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}

/* ---------------- section titles ---------------- */
.cura-section-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--cura-text);
    margin-bottom: 0.6rem;
}

/* generic card baseline -- applies to every bordered st.container, whether
   the key class lands on the bordered element itself or an ancestor of it */
[data-testid="stVerticalBlockBorderWrapper"],
.st-key-upload-card [data-testid="stVerticalBlockBorderWrapper"],
.st-key-result-success [data-testid="stVerticalBlockBorderWrapper"],
.st-key-result-rejected [data-testid="stVerticalBlockBorderWrapper"],
.st-key-result-failed [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    border: 1px solid var(--cura-border) !important;
    background: var(--cura-card-bg);
    box-shadow: 0 6px 24px rgba(143, 169, 255, 0.10);
    padding: 0.4rem;
}

/* tinted variants for result cards */
.st-key-result-rejected,
.st-key-result-failed {
    --tint-bg: linear-gradient(135deg, #FFF4ED, #FFE6DC);
    --tint-border: #F0C3A6;
}
.st-key-result-success {
    --tint-bg: linear-gradient(135deg, #FFFDFB, #F0E9FF);
    --tint-border: #D7C6FF;
}
.st-key-result-rejected,
.st-key-result-failed,
.st-key-result-success,
.st-key-result-rejected [data-testid="stVerticalBlockBorderWrapper"],
.st-key-result-failed [data-testid="stVerticalBlockBorderWrapper"],
.st-key-result-success [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--tint-bg) !important;
    border-color: var(--tint-border) !important;
}

/* ---------------- buttons ---------------- */
.stButton > button {
    background: linear-gradient(135deg, var(--cura-primary-soft), var(--cura-primary));
    color: #FFFFFF;
    border: none;
    border-radius: 14px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(143, 169, 255, 0.30);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(143, 169, 255, 0.40);
    color: #FFFFFF;
}
.stButton > button:disabled {
    background: #E8ECFB;
    color: #A8B4D6;
    box-shadow: none;
}

/* ---------------- upload dropzone ---------------- */
.cura-dropzone-icon {
    display: flex;
    justify-content: center;
    margin-top: 0.4rem;
}
.cura-dropzone-title {
    text-align: center;
    font-weight: 700;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}
[data-testid="stFileUploaderDropzone"] {
    background: #FFF8FB;
    border: 2px dashed var(--cura-lilac);
    border-radius: 16px;
    min-height: 120px;
    justify-content: center;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, var(--cura-primary-soft), var(--cura-primary)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(143, 169, 255, 0.30);
}
.cura-dropzone-caption {
    text-align: center;
    color: var(--cura-muted);
    font-size: 0.8rem;
    margin-top: 0.6rem;
}

/* small cropped preview thumbnail -- not the full image. Scoped to the
   upload card's own container (.st-key-thumb-wrap) rather than every
   [data-testid="stImage"] on the page: Streamlit's click-to-fullscreen
   view portals the image into a different overlay element, so scoping
   narrowly here means the fullscreen/zoomed view is unaffected and still
   opens at full size. */
.st-key-thumb-wrap [data-testid="stImage"] {
    display: flex;
    justify-content: center;
    margin-top: 0.8rem;
}
.st-key-thumb-wrap [data-testid="stImage"] img {
    width: 90px !important;
    height: 90px !important;
    object-fit: cover;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(143, 169, 255, 0.30);
    border: 1px solid var(--cura-border);
}

/* expander (privacy disclaimer) */
[data-testid="stExpander"] {
    border-radius: 16px;
    border: 1px solid var(--cura-border);
    background: #FFFFFF;
}

/* checkbox label readability */
.stCheckbox label p {
    font-weight: 600;
    color: var(--cura-text);
}

/* ---------------- Gemini summary rendering ---------------- */
.cura-summary-section-title {
    font-weight: 700;
    font-size: 1rem;
    color: var(--cura-text);
    margin: 1.1rem 0 0.5rem;
}
.cura-medicine-card {
    background: var(--cura-lavender);
    border: 1px solid var(--cura-border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
}
.cura-medicine-name {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.2rem;
}
.cura-medicine-meta {
    color: var(--cura-text-soft);
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
}
.cura-medicine-row {
    font-size: 0.88rem;
    margin-bottom: 0.2rem;
}
.cura-results-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
}
.cura-results-table th {
    text-align: left;
    background: var(--cura-lavender);
    color: var(--cura-primary-dark);
    padding: 0.55rem 0.7rem;
    font-weight: 700;
}
.cura-results-table td {
    padding: 0.55rem 0.7rem;
    border-bottom: 1px solid var(--cura-border);
}
.cura-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    white-space: nowrap;
}

/* ---------------- features grid ---------------- */
.cura-features-title {
    text-align: center;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 3rem 0 2rem;
    color: var(--cura-text);
}
.cura-feature-card {
    background: var(--cura-card-bg);
    border: 1px solid var(--cura-border);
    border-radius: 18px;
    padding: 1.3rem 1.4rem;
    height: 100%;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(143, 169, 255, 0.06);
}
.cura-feature-num {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: var(--cura-lavender);
    color: var(--cura-primary-dark);
    font-weight: 800;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.7rem;
}
.cura-feature-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.35rem;
}
.cura-feature-desc {
    font-size: 0.88rem;
    color: var(--cura-text-soft);
    line-height: 1.5;
}

/* ---------------- FAQ ---------------- */
.cura-faq-heading {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}
.cura-faq-sub {
    color: var(--cura-text-soft);
    font-size: 0.92rem;
}
details.cura-faq-item {
    border-bottom: 1px solid var(--cura-border);
    padding: 0.9rem 0;
}
details.cura-faq-item summary {
    cursor: pointer;
    font-weight: 700;
    font-size: 0.98rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    list-style: none;
    color: var(--cura-text);
}
details.cura-faq-item summary::-webkit-details-marker {
    display: none;
}
.cura-faq-toggle {
    font-size: 1.2rem;
    color: var(--cura-primary);
    transition: transform 0.15s ease;
    flex-shrink: 0;
    margin-left: 0.8rem;
}
details.cura-faq-item[open] .cura-faq-toggle {
    transform: rotate(45deg);
}
.cura-faq-answer {
    color: var(--cura-text-soft);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 0.6rem;
}

/* ---------------- footer ---------------- */
.cura-footer-dark {
    background: var(--cura-dark);
    border-radius: 24px;
    padding: 2.4rem 2.2rem 1.6rem;
    margin-top: 3.5rem;
    color: #C9D2E8;
}
.cura-footer-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 2rem;
}
.cura-footer-brand {
    font-size: 1.5rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.5rem;
}
.cura-footer-disclaimer {
    font-size: 0.8rem;
    color: #A7B0CC;
    max-width: 420px;
    line-height: 1.6;
}
.cura-footer-store-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8A93B0;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.cura-playstore-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: #0B0B0B;
    color: #FFFFFF;
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    border: 1px solid #3D4A6B;
    width: fit-content;
}
.cura-playstore-badge svg {
    flex-shrink: 0;
    display: block;
}
.cura-playstore-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
    line-height: 1.2;
}
.cura-playstore-text-small {
    font-size: 0.6rem;
    color: #C5CEE8;
}
.cura-playstore-text-big {
    font-size: 0.95rem;
    font-weight: 600;
}
.cura-footer-contact {
    color: #C9D2E8;
    font-weight: 600;
    font-size: 0.92rem;
}
.cura-footer-bottom {
    margin-top: 2rem;
    border-top: 1px solid #3D4A6B;
    padding-top: 1rem;
    font-size: 0.78rem;
    color: #8A93B0;
    text-align: center;
}

/* responsive tweaks for narrow screens */
@media (max-width: 768px) {
    .cura-hero-title { font-size: 2rem; }
    .cura-hero-sub { font-size: 0.95rem; }
    .cura-features-title, .cura-faq-heading { font-size: 1.5rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .cura-footer-grid { flex-direction: column; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ============================================================================
# END CUSTOM CSS
# ============================================================================

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cura-navbar">
        <div class="cura-navbar-logo">Cura</div>
        <div class="cura-navbar-tag">Medical Document Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cura-hero">
        <div class="cura-hero-badge">AI-Powered Summaries</div>
        <div class="cura-hero-title">
            Understand your medical documents <span class="cura-hero-accent">simply</span>
        </div>
        <div class="cura-hero-sub">
            Upload a prescription or lab report and get a clear, jargon-free summary in seconds.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load the Phase 3 pipeline (heavy models load once per server process --
# Python caches the module after the first import, so reruns are instant)
# ---------------------------------------------------------------------------
with st.spinner("Warming up Cura (first load only) ..."):
    from pipelines.phase3_pipeline import run_phase3_pipeline

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "analyzed_file_sig" not in st.session_state:
    st.session_state.analyzed_file_sig = None

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
with st.container(border=True, key="upload-card"):
    st.markdown(
        """
        <div class="cura-dropzone-icon">
            <svg width="38" height="38" viewBox="0 0 24 24" fill="none"
                 stroke="#2D3A5E" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <path d="M14 2v6h6"/>
                <path d="M12 18v-6"/>
                <path d="M9.5 14.5 12 12l2.5 2.5"/>
            </svg>
        </div>
        <div class="cura-dropzone-title">Drop your document here to upload</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload image",
        type=ALLOWED_UPLOAD_TYPES,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="cura-dropzone-caption">Accepts JPG, PNG, WEBP or HEIC (iPhone) &mdash; up to 20 MB</div>',
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        with st.container(key="thumb-wrap"):
            st.image(uploaded_file, width=90)

# ---------------------------------------------------------------------------
# Privacy disclaimer + consent (must be acknowledged before analyzing)
# ---------------------------------------------------------------------------
with st.expander("Privacy & disclaimer -- please read", expanded=True):
    st.markdown(
        "- This summary is AI-generated and may contain errors. "
        "Always cross-check with a qualified doctor before making any medical decisions.\n"
        "- Your image is processed temporarily to generate this summary and is not stored permanently.\n"
        "- By uploading, you consent to your document being analyzed by AI for summarization purposes.\n"
        "- This tool is for informational purposes only and is not a substitute for "
        "professional medical advice."
    )
consent_given = st.checkbox("I understand and consent to proceed", key="consent_given")

analyze_clicked = st.button(
    "Analyze Document",
    disabled=not (uploaded_file is not None and consent_given),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if analyze_clicked and uploaded_file is not None and consent_given:
    tmp_path = None
    try:
        with st.spinner("Reading your document ..."):
            suffix = Path(uploaded_file.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = run_phase3_pipeline(tmp_path)

        st.session_state.result = result
        st.session_state.analyzed_file_sig = (uploaded_file.name, uploaded_file.size)
    except Exception:
        traceback.print_exc()
        st.session_state.result = {"status": "error"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Clear a stale result if the user swapped in a different file without re-analyzing yet
current_sig = (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
if current_sig != st.session_state.analyzed_file_sig:
    st.session_state.result = None

# ---------------------------------------------------------------------------
# Gemini summary rendering -- gemini_summarizer_final.py's prompts ask for a
# structured JSON object (see PRESCRIPTION_PROMPT / REPORT_PROMPT); these
# helpers turn that JSON into the patient-friendly layout below instead of
# dumping the raw JSON text.
# ---------------------------------------------------------------------------
STATUS_BADGE_COLORS = {
    "NORMAL": ("#1F9D55", "#E3F6EA"),
    "LOW": ("#B7791F", "#FFF3DC"),
    "HIGH": ("#C0392B", "#FDE7E5"),
}
SEVERITY_BADGE_COLORS = {
    "MILD": ("#B7791F", "#FFF3DC"),
    "MODERATE": ("#C0631B", "#FFE7D2"),
    "SEVERE": ("#C0392B", "#FDE7E5"),
}


def _parse_gemini_json(summary_text):
    """Parse the JSON object Gemini returns, stripping ```json fences if present."""
    if not summary_text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", summary_text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _render_badge(label, color_map):
    if not label:
        return ""
    text_color, bg_color = color_map.get(label.upper(), ("#5B6680", "#F0E9FF"))
    return f'<span class="cura-badge" style="color:{text_color};background:{bg_color};">{escape(str(label))}</span>'


def _render_common_sections(data):
    """Do/don't, lifestyle tips and missing-info note -- shared by both doc types."""
    col_do, col_dont = st.columns(2)
    with col_do:
        if data.get("do_list"):
            st.markdown('<div class="cura-summary-section-title">Do</div>', unsafe_allow_html=True)
            for item in data["do_list"]:
                st.markdown(f"- {item}")
    with col_dont:
        if data.get("dont_list"):
            st.markdown('<div class="cura-summary-section-title">Don\'t</div>', unsafe_allow_html=True)
            for item in data["dont_list"]:
                st.markdown(f"- {item}")

    if data.get("lifestyle_tips"):
        st.markdown('<div class="cura-summary-section-title">Lifestyle tips</div>', unsafe_allow_html=True)
        for tip in data["lifestyle_tips"]:
            st.markdown(f"- {tip}")

    if data.get("missing_info_note"):
        st.warning(data["missing_info_note"])


def render_prescription_summary(data):
    if data.get("patient_summary"):
        st.markdown(data["patient_summary"])

    if data.get("diagnosis_explained"):
        st.markdown('<div class="cura-summary-section-title">About the diagnosis</div>', unsafe_allow_html=True)
        st.write(data["diagnosis_explained"])

    medicines = data.get("medicines") or []
    if medicines:
        st.markdown('<div class="cura-summary-section-title">Medicines</div>', unsafe_allow_html=True)
        for med in medicines:
            meta_bits = [b for b in (med.get("dosage"), med.get("frequency"), med.get("duration")) if b]
            detail_rows = "".join(
                f'<div class="cura-medicine-row"><b>{label}:</b> {escape(str(value))}</div>'
                for label, value in (
                    ("Used for", med.get("usage")),
                    ("Common side effects", med.get("common_side_effects")),
                    ("Alternatives", med.get("alternatives")),
                )
                if value
            )
            st.markdown(
                f"""
                <div class="cura-medicine-card">
                    <div class="cura-medicine-name">{escape(str(med.get('name') or 'Medicine'))}</div>
                    <div class="cura-medicine-meta">{escape(' · '.join(meta_bits))}</div>
                    {detail_rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_common_sections(data)


def render_report_summary(data):
    if data.get("patient_summary"):
        st.markdown(data["patient_summary"])

    if data.get("likely_condition"):
        st.markdown('<div class="cura-summary-section-title">Likely condition</div>', unsafe_allow_html=True)
        st.write(data["likely_condition"])

    test_results = data.get("test_results") or []
    if test_results:
        st.markdown('<div class="cura-summary-section-title">Test Results</div>', unsafe_allow_html=True)
        rows_html = "".join(
            f"""
            <tr>
                <td>{escape(str(t.get('test_name', '')))}</td>
                <td>{escape(str(t.get('value', '')))}</td>
                <td>{escape(str(t.get('reference_range', '')))}</td>
                <td>{_render_badge(t.get('status'), STATUS_BADGE_COLORS)}</td>
                <td>{_render_badge(t.get('severity'), SEVERITY_BADGE_COLORS)}</td>
            </tr>
            """
            for t in test_results
        )
        st.markdown(
            f"""
            <table class="cura-results-table">
                <thead>
                    <tr><th>Test</th><th>Value</th><th>Reference range</th><th>Status</th><th>Severity</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        meanings = [(t.get("test_name"), t.get("meaning")) for t in test_results if t.get("meaning")]
        if meanings:
            with st.expander("What each result means"):
                for name, meaning in meanings:
                    st.markdown(f"**{name}:** {meaning}")

    if data.get("overall_condition"):
        st.markdown('<div class="cura-summary-section-title">Overall takeaway</div>', unsafe_allow_html=True)
        st.write(data["overall_condition"])

    if data.get("specialist_suggestion"):
        st.info(data["specialist_suggestion"])

    _render_common_sections(data)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is not None:
    st.write("")  # small spacing before the results card
    status = result.get("status")
    doc_type = result.get("doc_type")

    if status == "rejected":
        with st.container(border=True, key="result-rejected"):
            st.markdown("#### This doesn't look like a medical document")
            st.write("Please upload a prescription or lab report.")

    elif status == "unreadable":
        with st.container(border=True, key="result-failed"):
            st.markdown("#### We couldn't read this clearly")
            st.write(
                "This can happen with blurry, handwritten, poorly lit, or partially "
                "captured photos. Please upload a clear, well-lit photo of the full "
                "document, or a digital copy if you have one."
            )

    elif status == "failed":
        with st.container(border=True, key="result-failed"):
            st.markdown("#### Something went wrong")
            st.write("Something went wrong, please try again.")

    elif status == "success" and doc_type in ("prescription", "report"):
        with st.container(border=True, key="result-success"):
            label = "Prescription Summary" if doc_type == "prescription" else "Lab Report Summary"
            st.markdown(f"#### {label}")
            summary_data = _parse_gemini_json(result.get("summary"))
            if summary_data is None:
                st.markdown(result.get("summary") or "_No summary available._")
            elif doc_type == "prescription":
                render_prescription_summary(summary_data)
            else:
                render_report_summary(summary_data)

    else:
        with st.container(border=True, key="result-failed"):
            st.markdown("#### Something went wrong")
            st.write("Something went wrong, please try again.")

# ---------------------------------------------------------------------------
# Features grid
# ---------------------------------------------------------------------------
st.markdown('<div class="cura-features-title">A complete solution for your medical documents</div>', unsafe_allow_html=True)

FEATURES = [
    ("01", "Accurate Text Extraction", "Advanced OCR reads prescriptions and lab reports, even from photographed documents."),
    ("02", "Smart Document Detection", "Automatically recognizes prescriptions and lab reports, and rejects non-medical uploads."),
    ("03", "Medical Entity Recognition", "Identifies drug names, dosages, test values and more using a biomedical AI model."),
    ("04", "Abnormality Detection", "Flags lab values outside the normal range so nothing important gets missed."),
    ("05", "Plain-Language Summaries", "Get a clear, jargon-free explanation of what your document actually means."),
    ("06", "Privacy-First Processing", "Your image is processed temporarily to generate a summary and is never stored."),
]

feat_cols = st.columns(3, gap="medium")
for i, (num, title, desc) in enumerate(FEATURES):
    with feat_cols[i % 3]:
        st.markdown(
            f"""
            <div class="cura-feature-card">
                <div class="cura-feature-num">{num}</div>
                <div class="cura-feature-title">{title}</div>
                <div class="cura-feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
st.write("")
st.write("")
faq_left, faq_right = st.columns([1, 1.4], gap="large")

with faq_left:
    st.markdown(
        """
        <div class="cura-faq-heading">Got questions about Cura?</div>
        <div class="cura-faq-sub">
            Here are some common questions about how Cura summarizes your medical documents.
        </div>
        """,
        unsafe_allow_html=True,
    )

FAQS = [
    ("Is Cura a substitute for medical advice?",
     "No. Cura provides AI-generated summaries for informational purposes only. "
     "Always cross-check with a qualified doctor before making any medical decisions."),
    ("Is my uploaded document stored?",
     "No. Your image is processed temporarily to generate the summary and is not stored permanently."),
    ("What file types can I upload?",
     "Cura accepts JPG, JPEG, PNG and WEBP images of prescriptions or lab reports, up to 20 MB."),
    ("What happens if I upload something that isn't medical?",
     "Cura's classifier detects non-medical uploads and will ask you to upload a valid "
     "prescription or lab report instead."),
]

with faq_right:
    faq_html = ""
    for question, answer in FAQS:
        faq_html += f"""
        <details class="cura-faq-item">
            <summary>{question}<span class="cura-faq-toggle">+</span></summary>
            <div class="cura-faq-answer">{answer}</div>
        </details>
        """
    st.markdown(faq_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cura-footer-dark">
        <div class="cura-footer-grid">
            <div>
                <div class="cura-footer-brand">Cura</div>
                <div class="cura-footer-disclaimer">
                    Cura provides AI-generated summaries for informational purposes only.
                    Always consult a healthcare professional.
                </div>
            </div>
            <div>
                <div class="cura-footer-store-label">Get the app</div>
                <div class="cura-playstore-badge">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="#FFFFFF">
                        <polygon points="6,3 6,21 20,12"/>
                    </svg>
                    <span class="cura-playstore-text">
                        <span class="cura-playstore-text-small">GET IT ON</span>
                        <span class="cura-playstore-text-big">Google Play</span>
                    </span>
                </div>
            </div>
            <div>
                <div class="cura-footer-store-label">Need help?</div>
                <div class="cura-footer-contact">curaaisupport@gmail.com</div>
            </div>
        </div>
        <div class="cura-footer-bottom">&copy; 2026 Cura. All rights reserved.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
