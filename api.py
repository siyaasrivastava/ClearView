"""ClearView web API + static UI.

    python serve.py            # http://127.0.0.1:8000

Wraps the same `clearview.process` pipeline the CLI and Streamlit app use.
Results are kept in a small in-memory cache so images can be served as plain URLs
(and downloaded) without re-running the pipeline. Nothing is written to disk.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from clearview import Config, process
from clearview.synthetic import (add_bubbles, add_vignette, corrupt, make_demo_dermoscopy,
                                 make_demo_endoscopy)
from clearview.seg import METHODS as SEG_METHODS_ALL, dice as seg_dice, status as seg_status
from clearview.utils import make_panel, outline_overlay

ROOT = Path(__file__).parent
DIST = ROOT / "web" / "dist"
MAX_UPLOAD = 50 * 1024 * 1024
CACHE_SIZE = 12

# BGR overlay colours. The same hexes are used by the UI legend.
ARTIFACTS = {
    "hair":    ("Hair",    (0, 200, 255)),
    "ink":     ("Ink",     (200, 60, 255)),
    "ruler":   ("Ruler",   (255, 200, 0)),
    "bubbles": ("Bubbles", (90, 230, 90)),
}
# Lesion-segmentation layers (BGR). Same hexes are used by the UI legend.
SEG_LAYERS = {
    "classical": ("Classical (Otsu)", (190, 190, 190)),
    "unet":      ("U-Net",            (110, 110, 255)),
    "medsam":    ("MedSAM",           (230, 200, 40)),
    "final":     ("Final mask",       (60, 230, 250)),
}
SEG_METHODS = set(SEG_METHODS_ALL)
MODALITIES = {"auto", "dermoscopy", "endoscopy"}
METHODS = {"telea", "ns", "criminisi"}

app = FastAPI(title="ClearView", version="2.1.0")
_cache: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()


# ----------------------------------------------------------------------------- helpers
def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _overlay_layer(mask: np.ndarray, bgr: tuple) -> np.ndarray:
    """Transparent PNG layer: the artifact colour where the mask is set."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    sel = mask > 0
    out[sel, 0], out[sel, 1], out[sel, 2], out[sel, 3] = bgr[0], bgr[1], bgr[2], 205
    return out


def _seg_layer(mask: np.ndarray, bgr: tuple) -> np.ndarray:
    """Transparent PNG layer for a lesion mask: light fill and a solid outline."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    sel = mask > 0
    out[sel] = (bgr[0], bgr[1], bgr[2], 70)
    cnts, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (bgr[0], bgr[1], bgr[2], 255), max(2, round(min(h, w) / 220)), cv2.LINE_AA)
    return out


def _seg_summary(res, base: str) -> dict | None:
    """JSON description of the lesion segmentation (None when it was not requested)."""
    seg = res.segmentation
    if seg is None:
        return None
    d = seg.to_dict()
    d["layers"] = [
        {"key": k, "label": SEG_LAYERS[k][0],
         "color": "#%02x%02x%02x" % (SEG_LAYERS[k][1][2], SEG_LAYERS[k][1][1], SEG_LAYERS[k][1][0]),
         "pixels": int((seg.layers[k] > 0).sum()), "layer": f"{base}/seg/{k}.png"}
        for k in SEG_LAYERS if k in seg.layers
    ]
    d["mask"] = f"{base}/lesion_mask.png"
    orig = res.segmentation_original
    d["dice_vs_uncleaned"] = None if orig is None else round(seg_dice(seg.mask, orig.mask), 4)
    return d


def _colored_overlay(img: np.ndarray, masks: dict, alpha: float = 0.75) -> np.ndarray:
    out = img.copy()
    for key, (_, bgr) in ARTIFACTS.items():
        sel = masks.get(key) is not None and masks[key] > 0
        if np.any(sel):
            out[sel] = (alpha * np.array(bgr) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out


def _sample(kind: str) -> np.ndarray:
    """Synthetic demo images (no patient data). Same recipe as `run.py --demo`."""
    if kind == "dermoscopy":
        img, _ = make_demo_dermoscopy(384, seed=3)
        return corrupt(img, seed=3, n_hair=20, ruler=True, ink=True, n_bubbles=2)["corrupted"]
    if kind == "endoscopy":
        rng = np.random.default_rng(1)
        img = add_vignette(make_demo_endoscopy(384, seed=2), 0.4)
        return add_bubbles(img, rng, 4)[0]
    raise HTTPException(404, "Unknown sample. Use 'dermoscopy' or 'endoscopy'.")


def _store(entry: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    with _lock:
        _cache[rid] = entry
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return rid


def _get(rid: str) -> dict:
    with _lock:
        entry = _cache.get(rid)
        if entry is not None:
            _cache.move_to_end(rid)
    if entry is None:
        raise HTTPException(404, "This result has expired. Process the image again.")
    return entry


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/segmentation/status")
def segmentation_status():
    """Which lesion-segmentation methods can run on this machine, and why not if one cannot."""
    st = seg_status()
    st.pop("unet_weights", None)          # do not leak server file paths to the browser
    st.pop("medsam_weights", None)
    return st


@app.get("/api/samples/{kind}.png")
def sample_png(kind: str):
    return Response(_png(_sample(kind)), media_type="image/png")


@app.post("/api/process")
def process_image(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    options: str = Form("{}"),
):
    try:
        opts = json.loads(options) or {}
    except json.JSONDecodeError:
        raise HTTPException(400, "Options must be valid JSON.")

    if file is not None and file.filename:
        data = file.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "This file is larger than 50 MB. Use a smaller image.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "This file could not be read as an image. Upload a JPG, PNG, BMP or TIFF file.")
        name = Path(file.filename).stem or "image"
    elif sample:
        img, name = _sample(sample), f"sample_{sample}"
    else:
        raise HTTPException(400, "Send an image file or choose a sample.")

    modality = opts.get("modality", "auto")
    method = opts.get("inpaint_method", "telea")
    if modality not in MODALITIES or method not in METHODS:
        raise HTTPException(400, "Unknown image type or fill-in method.")
    hair = bool(opts.get("hair", True))
    seg_method = opts.get("segment", "none")
    if seg_method not in SEG_METHODS:
        raise HTTPException(400, "Unknown lesion segmentation method.")
    box = opts.get("box")
    if box is not None:
        try:
            box = tuple(float(v) for v in box)
            assert len(box) == 4 and all(0.0 <= v <= 1.0 for v in box)
        except (TypeError, ValueError, AssertionError):
            raise HTTPException(400, "A box prompt needs four numbers between 0 and 1: x0, y0, x1, y1.")
    # Model weights are never taken from the request (that would let a client load arbitrary files);
    # they come from CLEARVIEW_UNET_WEIGHTS / CLEARVIEW_MEDSAM_WEIGHTS or the models/ folder.
    cfg = Config(
        modality=modality, inpaint_method=method,
        correct_illumination=bool(opts.get("correct_illumination", True)),
        clahe=bool(opts.get("clahe", False)),
        hair=hair, light_hair=bool(opts.get("light_hair", True)) and hair,
        ink=bool(opts.get("ink", True)), ruler=bool(opts.get("ruler", True)),
        bubbles=bool(opts.get("bubbles", True)),
        max_side=int(min(max(int(opts.get("max_side", 512)), 128), 2048)),
        segment=seg_method, seg_box=box, seg_compare_original=bool(opts.get("seg_compare_original", False)),
    )
    try:
        res = process(img, cfg)
    except Exception as exc:  # keep the server alive on odd inputs
        raise HTTPException(422, f"Processing failed: {exc}. Try another image or turn off some detectors.")

    rid = _store({"res": res, "name": name})
    h, w = res.original.shape[:2]
    base = f"/api/result/{rid}"
    return JSONResponse({
        "id": rid,
        "name": name,
        "width": w, "height": h,
        "modality": res.modality,
        "inpaint_skipped": res.inpaint_skipped,
        "before": res.before.to_dict(),
        "after": res.after.to_dict(),
        "artifacts": [
            {"key": k, "label": lab, "color": "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0]),
             "pixels": int((res.masks[k] > 0).sum()), "layer": f"{base}/layer/{k}.png"}
            for k, (lab, bgr) in ARTIFACTS.items() if k in res.masks
        ],
        "images": {
            "original": f"{base}/original.png",
            "cleaned": f"{base}/cleaned.png",
            "mask": f"{base}/mask.png",
            "panel": f"{base}/panel.png",
        },
        "report": f"{base}/report.json",
        "segmentation": _seg_summary(res, base),
        "segmentation_error": res.info.get("segmentation_error"),
    })


def _png_response(data: bytes, filename: str | None = None) -> Response:
    headers = {"Cache-Control": "private, max-age=3600"}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(data, media_type="image/png", headers=headers)


@app.get("/api/result/{rid}/{kind}.png")
def result_image(rid: str, kind: str, download: bool = False):
    e = _get(rid)
    res, name = e["res"], e["name"]
    if kind == "original":
        data = _png(res.original)
    elif kind == "cleaned":
        data = _png(res.cleaned)
    elif kind == "mask":
        data = _png(res.fused_mask * 255)
    elif kind == "lesion_mask":
        if res.segmentation is None:
            raise HTTPException(404, "No lesion segmentation for this result.")
        data = _png(res.segmentation.mask * 255)
    elif kind == "lesion_overlay":
        if res.segmentation is None:
            raise HTTPException(404, "No lesion segmentation for this result.")
        data = _png(outline_overlay(res.cleaned, res.segmentation.mask))
    elif kind == "panel":
        panel = make_panel([(res.original, f"input ({res.modality})"),
                            (_colored_overlay(res.original, res.masks), "detected artifacts"),
                            (res.cleaned, f"cleaned [{res.after.decision}]")])
        data = _png(panel)
    else:
        raise HTTPException(404, "Unknown image.")
    return _png_response(data, f"{name}_{kind}.png" if download else None)


@app.get("/api/result/{rid}/layer/{key}.png")
def result_layer(rid: str, key: str):
    res = _get(rid)["res"]
    if key not in ARTIFACTS or key not in res.masks:
        raise HTTPException(404, "Unknown artifact layer.")
    return _png_response(_png(_overlay_layer(res.masks[key], ARTIFACTS[key][1])))


@app.get("/api/result/{rid}/seg/{key}.png")
def result_seg_layer(rid: str, key: str):
    seg = _get(rid)["res"].segmentation
    if seg is None or key not in SEG_LAYERS or key not in seg.layers:
        raise HTTPException(404, "Unknown segmentation layer.")
    return _png_response(_png(_seg_layer(seg.layers[key], SEG_LAYERS[key][1])))


@app.get("/api/result/{rid}/report.json")
def result_report(rid: str, download: bool = False):
    e = _get(rid)
    res, name = e["res"], e["name"]
    report = {
        "image": name, "modality": res.modality, "inpaint_skipped": res.inpaint_skipped,
        "artifact_pixels": {k: int(v.sum()) for k, v in res.masks.items()},
        "before": res.before.to_dict(), "after": res.after.to_dict(),
        "ruler": res.info.get("ruler"), "bubbles_found": len(res.info.get("bubbles", [])),
    }
    if res.segmentation is not None:
        report["segmentation"] = res.segmentation.to_dict()
    headers = {"Content-Disposition": f'attachment; filename="{name}_report.json"'} if download else {}
    return Response(json.dumps(report, indent=2, default=str), media_type="application/json", headers=headers)


# ----------------------------------------------------------------------------- UI
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_ui():
        return JSONResponse({"error": "UI is not built. Run `cd web && npm install && npm run build`."}, status_code=503)
