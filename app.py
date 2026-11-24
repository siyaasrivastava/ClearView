"""ClearView GUI:  streamlit run app.py"""
import base64
import json
import os

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from clearview import Config, process
from clearview.synthetic import (add_bubbles, add_vignette, corrupt, make_demo_dermoscopy,
                                 make_demo_endoscopy)
from clearview.seg import dice as seg_dice, status as seg_status
from clearview.utils import make_panel, outline_overlay

st.set_page_config(page_title="ClearView", page_icon="🔬", layout="wide")

# ----------------------------------------------------------------------------
# Look and feel: pale clinical chrome around a neutral grey image stage
# ----------------------------------------------------------------------------
FONT_URL = "https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&display=swap"
st.markdown(f"""
<style>
@import url('{FONT_URL}');
html, body, .stApp, .stMarkdown, p, label, li, td, th, button, input, textarea,
[data-baseweb="select"] div, [data-testid="stWidgetLabel"] p {{
    font-family: 'Hanken Grotesk', system-ui, sans-serif;
}}
/* keep Streamlit's icon font for toggles, uploader and expander arrows */
[data-testid="stIconMaterial"], span[class*="material"], .material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
#MainMenu, footer, .stAppDeployButton {{visibility: hidden;}}
.block-container {{padding-top: 2.2rem; max-width: 1180px;}}
h1, h2, h3 {{letter-spacing: -0.01em; font-weight: 650;}}

.cv-brand {{display:block; margin-bottom:.15rem;}}
.cv-brand .cv-title {{margin:0; padding:0; font-size:2.1rem; font-weight:700; letter-spacing:-0.02em;
                      color:#0E5A6B; line-height:1.3; padding-top:.2rem;}}
.cv-brand span {{display:block; color:#4A555D; font-size:1rem; margin-top:.1rem;}}
.cv-note {{color:#5B6770; font-size:.88rem; margin:0 0 1.4rem 0;}}

.cv-stage {{background:#23282D; border-radius:6px; padding:10px; display:flex;
            align-items:center; justify-content:center; min-height:120px;}}
.cv-stage img {{max-width:100%; max-height:520px; object-fit:contain; display:block;}}
.cv-cap {{font-size:.86rem; color:#3B464E; margin:.45rem 0 0 .1rem; font-weight:500;}}
.cv-cap small {{color:#6B7780; font-weight:400;}}

.cv-verdict {{border-left:6px solid var(--c); background:#fff; border-radius:6px;
              padding:1rem 1.25rem; display:flex; gap:1.4rem; align-items:center;
              box-shadow:0 0 0 1px #DDE2E6 inset; margin-bottom:1.1rem;}}
.cv-verdict.accept {{--c:#1E7B4F;}}
.cv-verdict.clean {{--c:#A66F00;}}
.cv-verdict.recapture {{--c:#B3372D;}}
.cv-score {{font-size:2.6rem; font-weight:700; line-height:1; color:var(--c);
            font-variant-numeric: tabular-nums;}}
.cv-score small {{font-size:1rem; color:#6B7780; font-weight:500; margin-left:.15rem;}}
.cv-decision {{font-size:1.15rem; font-weight:650; color:#1B2329;}}
.cv-sub {{color:#4A555D; font-size:.95rem; margin-top:.15rem;}}

table.cv-table {{width:100%; border-collapse:collapse; font-size:.93rem;
                 font-variant-numeric: tabular-nums;}}
table.cv-table th {{text-align:left; font-weight:600; color:#4A555D; padding:.5rem .6rem;
                    border-bottom:1px solid #CBD2D8;}}
table.cv-table td {{padding:.55rem .6rem; border-bottom:1px solid #E3E7EA;}}
table.cv-table td.num, table.cv-table th.num {{text-align:right;}}
.cv-hint {{color:#6B7780; font-size:.82rem;}}
.better {{color:#1E7B4F; font-weight:600;}}
.worse {{color:#B3372D; font-weight:600;}}
.same {{color:#6B7780;}}

.cv-legend {{display:flex; flex-wrap:wrap; gap:.5rem; margin:.6rem 0 .2rem 0;}}
.cv-chip {{display:inline-flex; align-items:center; gap:.45rem; background:#fff;
           border:1px solid #D3D9DE; border-radius:999px; padding:.22rem .75rem; font-size:.88rem;}}
.cv-chip i {{width:.8rem; height:.8rem; border-radius:50%; display:inline-block;}}
.cv-chip b {{font-weight:600; font-variant-numeric: tabular-nums;}}
.cv-chip.off {{color:#8A949B; background:transparent; border-style:dashed;}}

section[data-testid="stSidebar"] {{border-right:1px solid #DDE2E6;}}
section[data-testid="stSidebar"] h2 {{font-size:1.05rem;}}
.cv-side-h {{font-weight:650; font-size:.92rem; margin:1rem 0 .15rem 0; color:#1B2329;}}
:focus-visible {{outline:3px solid #0E5A6B !important; outline-offset:2px;}}
@media (prefers-reduced-motion: reduce) {{* {{transition:none !important; animation:none !important;}}}}
</style>
""", unsafe_allow_html=True)

# BGR colours used for the artifact overlays (and the matching legend dots)
ARTIFACT_STYLE = {
    "hair":    ("Hair",    (0, 200, 255)),
    "ink":     ("Ink",     (200, 60, 255)),
    "ruler":   ("Ruler",   (255, 200, 0)),
    "bubbles": ("Bubbles", (90, 230, 90)),
}
DECISIONS = {
    "Accept":     ("accept",    "Ready to use as it is"),
    "Clean":      ("clean",     "Usable after cleaning"),
    "Re-capture": ("recapture", "Re-capture recommended"),
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def data_uri(img, ext=".png", quality=92):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode()}"


def stage(img, alt, caption="", sub=""):
    """A picture on the neutral grey stage, with an optional caption."""
    cap = f'<div class="cv-cap">{caption} <small>{sub}</small></div>' if caption else ""
    st.markdown(f'<div class="cv-stage"><img src="{data_uri(img, ".jpg")}" alt="{alt}"></div>{cap}',
                unsafe_allow_html=True)


def colored_overlay(img, masks, alpha=0.75):
    out = img.copy()
    for key, (_, bgr) in ARTIFACT_STYLE.items():
        m = masks.get(key)
        if m is None:
            continue
        sel = m > 0
        if sel.any():
            out[sel] = (alpha * np.array(bgr) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def comparison_slider(before, after, height=480):
    """Draggable before/after viewer. Keyboard accessible through a range input."""
    h, w = before.shape[:2]
    html = f"""
<style>
@import url('{FONT_URL}');
body{{margin:0;font-family:'Hanken Grotesk',system-ui,sans-serif;}}
.wrap{{background:#23282D;border-radius:6px;padding:10px;height:{height - 20}px;display:flex;
      align-items:center;justify-content:center;box-sizing:border-box;}}
.frame{{position:relative;aspect-ratio:{w}/{h};max-width:100%;max-height:100%;height:100%;user-select:none;}}
.frame img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;}}
#top{{clip-path:inset(0 50% 0 0);}}
.tag{{position:absolute;top:8px;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:600;
     background:rgba(20,24,28,.78);color:#fff;pointer-events:none;}}
#l{{left:8px}} #r{{right:8px}}
.line{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;
      box-shadow:0 0 0 1px rgba(0,0,0,.35);pointer-events:none;}}
.knob{{position:absolute;top:50%;left:50%;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;
      background:#fff;box-shadow:0 1px 5px rgba(0,0,0,.5);pointer-events:none;display:flex;
      align-items:center;justify-content:center;color:#23282D;font-size:15px;font-weight:700;}}
input[type=range]{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0;}}
input[type=range]:focus-visible ~ .knob{{outline:3px solid #5CC8E0;outline-offset:2px;}}
</style>
<div class="wrap"><div class="frame">
  <img src="{data_uri(after, '.jpg')}" alt="Cleaned image">
  <img id="top" src="{data_uri(before, '.jpg')}" alt="Input image">
  <span class="tag" id="l">Input</span><span class="tag" id="r">Cleaned</span>
  <div class="line" id="line"></div><div class="knob" id="knob">&#8596;</div>
  <input id="s" type="range" min="0" max="100" value="50" aria-label="Compare input and cleaned image">
</div></div>
<script>
const s=document.getElementById('s'),t=document.getElementById('top'),
      l=document.getElementById('line'),k=document.getElementById('knob');
function upd(){{const v=s.value;t.style.clipPath='inset(0 '+(100-v)+'% 0 0)';
  l.style.left=v+'%';k.style.left=v+'%';}}
s.addEventListener('input',upd);upd();
</script>"""
    if hasattr(st, "iframe"):          # newer Streamlit; components.html is being retired
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


@st.cache_data(show_spinner=False, max_entries=6)
def run_pipeline(data: bytes, cfg_items: tuple):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return process(img, Config(**dict(cfg_items)))


@st.cache_data(show_spinner=False)
def sample_image(kind: str) -> bytes:
    """Synthetic demo images (no patient data), same recipe as `run.py --demo`."""
    if kind == "dermoscopy":
        img, _ = make_demo_dermoscopy(384, seed=3)
        img = corrupt(img, seed=3, n_hair=20, ruler=True, ink=True, n_bubbles=2)["corrupted"]
    else:
        rng = np.random.default_rng(1)
        img = add_vignette(make_demo_endoscopy(384, seed=2), 0.4)
        img, _ = add_bubbles(img, rng, 4)
    return cv2.imencode(".png", img)[1].tobytes()


def verdict(res):
    b, a = res.before, res.after
    cls, headline = DECISIONS.get(a.decision, ("clean", a.decision))
    delta = a.score - b.score
    if res.inpaint_skipped:
        sub = ("Too much of the image is covered by artifacts to rebuild reliably, so nothing was "
               "inpainted. Lighting correction was still applied.")
    elif abs(delta) < 1:
        sub = f"Score unchanged from {b.score:.0f} before cleaning ({b.decision})."
    else:
        word = "up" if delta > 0 else "down"
        sub = f"Score is {word} from {b.score:.0f} before cleaning ({b.decision})."
    st.markdown(f"""
<div class="cv-verdict {cls}">
  <div class="cv-score">{a.score:.0f}<small>/ 100</small></div>
  <div><div class="cv-decision">{headline}</div><div class="cv-sub">{sub}</div></div>
</div>""", unsafe_allow_html=True)


def metrics_table(res):
    b, a = res.before, res.after
    # (label, before, after, format, higher_is_better or None if neutral, hint)
    # Sharpness and contrast are measured on edges, and hair, ink and ruler edges inflate
    # them, so a drop after cleaning is expected and is not judged good or bad.
    rows = [
        ("Sharpness", b.sharpness, a.sharpness, "{:.0f}", None, "artifact edges inflate this, so it often drops"),
        ("Lighting unevenness", b.illum_cv, a.illum_cv, "{:.3f}", False, "lower is better"),
        ("Contrast", b.contrast, a.contrast, "{:.1f}", None, "artifacts add contrast, so it often drops"),
        ("Artifact coverage", 100 * b.artifact_cover, 100 * a.artifact_cover, "{:.1f}%", False, "lower is better"),
    ]
    body = ""
    for label, x, y, fmt, hib, hint in rows:
        d = y - x
        tol = 0.02 * max(abs(x), 1e-6)
        if abs(d) <= tol:
            cls, word = "same", "no change"
        elif hib is None:
            cls, word = "same", "higher" if d > 0 else "lower"
        elif (d > 0) == hib:
            cls, word = "better", "better"
        else:
            cls, word = "worse", "worse"
        body += (f"<tr><td>{label}<div class='cv-hint'>{hint}</div></td>"
                 f"<td class='num'>{fmt.format(x)}</td><td class='num'>{fmt.format(y)}</td>"
                 f"<td class='num {cls}'>{word}</td></tr>")
    st.markdown(f"""<table class="cv-table"><thead><tr><th>Measure</th><th class="num">Before</th>
<th class="num">After</th><th class="num">Change</th></tr></thead><tbody>{body}</tbody></table>""",
                unsafe_allow_html=True)


def legend(res):
    chips = ""
    for key, (label, bgr) in ARTIFACT_STYLE.items():
        m = res.masks.get(key)
        n = int((m > 0).sum()) if m is not None else 0
        r, g, b = bgr[2], bgr[1], bgr[0]
        off = "" if n else " off"
        chips += (f'<span class="cv-chip{off}"><i style="background:rgb({r},{g},{b})"></i>'
                  f'{label} <b>{n:,} px</b></span>')
    st.markdown(f'<div class="cv-legend">{chips}</div>', unsafe_allow_html=True)


def report_dict(res, name):
    return {
        "image": name, "modality": res.modality, "inpaint_skipped": res.inpaint_skipped,
        "artifact_pixels": {k: int(v.sum()) for k, v in res.masks.items()},
        "before": res.before.to_dict(), "after": res.after.to_dict(),
        "ruler": res.info.get("ruler"), "bubbles_found": len(res.info.get("bubbles", [])),
        **({"segmentation": res.segmentation.to_dict()} if res.segmentation is not None else {}),
    }


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


# ----------------------------------------------------------------------------
# Sidebar: settings, grouped by what they change
# ----------------------------------------------------------------------------
MODALITIES = {"Detect automatically": "auto", "Dermoscopy (skin)": "dermoscopy", "Endoscopy": "endoscopy"}
METHODS = {"Fast": "telea", "Smooth": "ns", "Texture-aware": "criminisi"}

with st.sidebar:
    st.header("Settings")
    st.markdown('<div class="cv-side-h">Image type</div>', unsafe_allow_html=True)
    modality = MODALITIES[st.selectbox("Image type", list(MODALITIES), label_visibility="collapsed",
                                       help="Automatic mode looks for the green position inset "
                                            "used by endoscopes. Choose explicitly if you know.")]

    st.markdown('<div class="cv-side-h">Remove these artifacts</div>', unsafe_allow_html=True)
    hair = st.checkbox("Hair", True)
    light = st.checkbox("Include light hair", True, disabled=not hair)
    ink = st.checkbox("Ink marks", True)
    ruler = st.checkbox("Ruler", True, help="Axis-aligned rulers only.")
    bubbles = st.checkbox("Air bubbles", True)

    st.markdown('<div class="cv-side-h">Fill-in method</div>', unsafe_allow_html=True)
    method = METHODS[st.selectbox("Fill-in method", list(METHODS), label_visibility="collapsed",
                                  help="How removed areas are reconstructed. Fast and Smooth take "
                                       "a fraction of a second; Texture-aware takes several seconds.")]

    st.markdown('<div class="cv-side-h">Lighting</div>', unsafe_allow_html=True)
    illum = st.checkbox("Even out lighting", True, help="Homomorphic filter; fixes vignetting and glare gradients.")
    clahe = st.checkbox("Boost local contrast", False, help="CLAHE, applied after lighting correction.")

    st.markdown('<div class="cv-side-h">Lesion segmentation</div>', unsafe_allow_html=True)
    _st = seg_status()
    SEG_LABELS = {"none": "Off", "classical": "Classical threshold", "unet": "U-Net", "medsam": "MedSAM",
                  "hybrid": "U-Net + MedSAM"}
    seg_choices = [k for k in SEG_LABELS if _st["methods"].get(k)]
    segment = st.selectbox("Lesion segmentation", seg_choices, format_func=SEG_LABELS.get,
                           label_visibility="collapsed",
                           help="Outlines the lesion on the cleaned image. U-Net: a model trained on your data. "
                                "MedSAM: a general medical model prompted with a box. U-Net + MedSAM: the U-Net "
                                "finds the lesion, MedSAM refines it, and the two are compared.")
    seg_cmp = st.checkbox("Compare with uncleaned image", False, disabled=segment == "none",
                          help="Segments the original too, to show what cleaning changed. Runs the segmentation twice.")
    _missing = [SEG_LABELS[k] for k in ("unet", "medsam", "hybrid") if not _st["methods"].get(k)]
    if _missing:
        st.caption("Not available here: " + ", ".join(_missing) + ". See the README, section Lesion segmentation.")

    st.markdown('<div class="cv-side-h">Speed</div>', unsafe_allow_html=True)
    max_side = st.slider("Longest side (px)", 256, 1024, 512, 64,
                         help="Larger images are scaled down before processing. Higher values keep more detail but run slower.")

# ----------------------------------------------------------------------------
# Header + input
# ----------------------------------------------------------------------------
st.markdown('<div class="cv-brand"><div class="cv-title">ClearView</div>'
            '<span>Remove hair, ink, rulers and bubbles from dermoscopy and endoscopy images, and score the result.</span></div>'
            '<p class="cv-note">Preprocessing and quality check only. Not a diagnostic tool.</p>',
            unsafe_allow_html=True)

up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                      help="JPG, PNG, BMP or TIFF. The image is processed in memory and not stored.")

if up is not None:
    st.session_state.pop("sample", None)
    data, name = up.getvalue(), os.path.splitext(up.name)[0]
elif st.session_state.get("sample"):
    kind = st.session_state["sample"]
    data, name = sample_image(kind), f"sample_{kind}"
else:
    data, name = None, None

if data is None:
    st.markdown("**No image at hand?** Try a synthetic sample (no patient data):")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Dermoscopy sample", use_container_width=True):
        st.session_state["sample"] = "dermoscopy"
        st.rerun()
    if c2.button("Endoscopy sample", use_container_width=True):
        st.session_state["sample"] = "endoscopy"
        st.rerun()
    st.stop()

if up is None:
    note, btn = st.columns([4, 1])
    note.caption(f"Showing the built-in {st.session_state['sample']} sample. Upload an image above to use your own.")
    if btn.button("Clear sample", use_container_width=True):
        st.session_state.pop("sample", None)
        st.rerun()

# ----------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------
cfg_items = tuple(sorted(dict(
    modality=modality, inpaint_method=method, correct_illumination=illum, clahe=clahe,
    hair=hair, light_hair=light and hair, ink=ink, ruler=ruler, bubbles=bubbles, max_side=max_side,
    segment=segment, seg_compare_original=seg_cmp,
).items()))

try:
    with st.spinner("Cleaning image…"):
        res = run_pipeline(data, cfg_items)
except Exception as exc:  # keep the app alive on odd inputs
    st.error(f"Processing failed: {exc}. Try a different image or turn off some artifact detectors.")
    st.stop()
if res is None:
    st.error("This file could not be read as an image. Upload a JPG, PNG, BMP or TIFF file.")
    st.stop()

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
verdict(res)
if res.inpaint_skipped:
    st.warning("Artifact coverage is above the safe limit, so the image was not inpainted. "
               "Consider re-capturing it.")

mode = st.radio("View", ["Before / after slider", "Side by side", "Detected artifacts"],
                horizontal=True, label_visibility="collapsed")
modality_label = "dermoscopy" if res.modality == "dermoscopy" else "endoscopy"

if mode == "Before / after slider":
    comparison_slider(res.original, res.cleaned)
    st.caption(f"Drag across the image, or focus it and use the arrow keys. Detected type: {modality_label}.")
elif mode == "Side by side":
    a, b, c = st.columns(3)
    with a:
        stage(res.original, "Input image", "Input", modality_label)
    with b:
        stage(colored_overlay(res.original, res.masks), "Detected artifacts", "Detected artifacts")
    with c:
        stage(res.cleaned, "Cleaned image", "Cleaned")
    legend(res)
else:
    stage(colored_overlay(res.original, res.masks), "Input image with detected artifacts highlighted")
    legend(res)

st.markdown("### Image quality")
left, right = st.columns([3, 2], gap="large")
with left:
    metrics_table(res)
with right:
    with st.expander("Per-artifact masks"):
        keys = list(res.masks)
        cols = st.columns(2)
        for i, k in enumerate(keys):
            with cols[i % 2]:
                stage(res.masks[k] * 255, f"{k} mask", ARTIFACT_STYLE.get(k, (k,))[0],
                      f"{int(res.masks[k].sum()):,} px")

seg = res.segmentation
if res.info.get("segmentation_error"):
    st.warning(f"Lesion outline failed: {res.info['segmentation_error']} The cleaned image above is still valid.")
if seg is not None:
    st.markdown("### Lesion outline")
    USED = {"none": "nothing", "classical": "a classical threshold", "unet": "the U-Net", "medsam": "MedSAM",
            "hybrid": "the U-Net and MedSAM together"}
    if seg.used == "none":
        st.info("No lesion outline could be made.")
    else:
        extra = f" (you asked for {USED[seg.requested]}, which was not available)" if seg.used != seg.requested else ""
        st.markdown(f"Outlined with {USED[seg.used]}{extra}.")
    lc, rc = st.columns([3, 2], gap="large")
    with lc:
        stage(outline_overlay(res.cleaned, seg.mask), "Cleaned image with the lesion outlined", "Lesion outline",
              seg.used)
        others = [k for k in ("classical", "unet", "medsam") if k in seg.layers]
        if len(others) > 1:
            with st.expander("Compare outlines"):
                cols = st.columns(len(others))
                for c, k in zip(cols, others):
                    with c:
                        stage(outline_overlay(res.cleaned, seg.layers[k]), f"{k} outline", k)
    with rc:
        m1, m2 = st.columns(2)
        m1.metric("Lesion area", f"{100 * seg.area_frac:.1f}%", help="Share of the usable field of view.")
        if seg.agreement is not None:
            m2.metric("U-Net vs MedSAM", f"{seg.agreement:.2f}", help="Dice overlap of the two outlines.")
        if seg.stability is not None:
            m1.metric("Box stability", f"{seg.stability:.2f}", help="Dice between MedSAM masks for slightly shifted boxes.")
        if res.segmentation_original is not None:
            m2.metric("Overlap with uncleaned", f"{seg_dice(seg.mask, res.segmentation_original.mask):.2f}",
                      help="Outline on the cleaned vs the original image.")
        (st.warning if seg.review else st.success)(
            "Worth a look before you rely on this outline." if seg.review else "No warnings from the automatic checks.")
        for n in seg.notes:
            st.caption("• " + n)
        st.caption("Automatic outline for preprocessing and review. Not a diagnosis; the checks are heuristics.")

st.markdown("### Export")
d1, d2, d3, d4 = st.columns(4)
if seg is not None:
    st.download_button("Lesion mask", png_bytes(seg.mask * 255), f"{name}_lesion_mask.png", "image/png")
d1.download_button("Cleaned image", png_bytes(res.cleaned), f"{name}_cleaned.png", "image/png",
                   type="primary", use_container_width=True)
d2.download_button("Artifact mask", png_bytes(res.fused_mask * 255), f"{name}_mask.png", "image/png",
                   use_container_width=True)
panel = make_panel([(res.original, f"input ({res.modality})"),
                    (colored_overlay(res.original, res.masks), "detected artifacts"),
                    (res.cleaned, f"cleaned [{res.after.decision}]")])
d3.download_button("Comparison panel", png_bytes(panel), f"{name}_panel.png", "image/png",
                   use_container_width=True)
d4.download_button("Quality report (JSON)", json.dumps(report_dict(res, name), indent=2, default=str),
                   f"{name}_report.json", "application/json", use_container_width=True)
