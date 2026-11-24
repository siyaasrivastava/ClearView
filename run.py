"""Command-line interface.

    python run.py path/to/image.jpg                      # one image
    python run.py path/to/folder --out results           # whole folder
    python run.py img.png --modality endoscopy --inpaint ns
    python run.py --demo                                  # built-in synthetic demo
"""
import argparse
import dataclasses
import json
import os

import cv2

from clearview import Config, process
from clearview.synthetic import corrupt, make_demo_dermoscopy, make_demo_endoscopy, add_bubbles, add_vignette
from clearview.seg import box_to_norm
from clearview.utils import imread, imwrite, list_images, make_panel, outline_overlay, overlay_mask
import numpy as np


def run_one(path_or_img, name, args, cfg):
    img = imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    if args.box:                                   # pixel box of the *input* image -> normalised
        h, w = img.shape[:2]
        cfg = dataclasses.replace(cfg, seg_box=box_to_norm(tuple(args.box), w, h))
    res = process(img, cfg)
    out = args.out
    imwrite(os.path.join(out, f"{name}_cleaned.png"), res.cleaned)
    imwrite(os.path.join(out, f"{name}_mask.png"), res.fused_mask * 255)
    tiles = [
        (res.original, f"input ({res.modality})"),
        (overlay_mask(res.original, res.fused_mask), "detected artifacts"),
        (res.cleaned, f"cleaned [{res.after.decision}]"),
    ]
    seg = res.segmentation
    if seg is not None:
        imwrite(os.path.join(out, f"{name}_lesion.png"), seg.mask * 255)
        tiles.append((outline_overlay(res.cleaned, seg.mask), f"lesion [{seg.used}]"))
    panel = make_panel(tiles)
    imwrite(os.path.join(out, f"{name}_panel.png"), panel)
    report = {
        "image": name, "modality": res.modality, "inpaint_skipped": res.inpaint_skipped,
        "artifact_pixels": {k: int(v.sum()) for k, v in res.masks.items()},
        "before": res.before.to_dict(), "after": res.after.to_dict(),
        "ruler": res.info.get("ruler"), "bubbles_found": len(res.info.get("bubbles", [])),
    }
    if seg is not None:
        report["segmentation"] = seg.to_dict()
        if res.segmentation_original is not None:
            from clearview.seg import dice
            report["segmentation"]["dice_vs_uncleaned"] = round(dice(seg.mask, res.segmentation_original.mask), 4)
    if "segmentation_error" in res.info:
        report["segmentation_error"] = res.info["segmentation_error"]
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="ClearView: artifact removal + quality scoring")
    ap.add_argument("input", nargs="?", help="image file or folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--modality", default="auto", choices=["auto", "dermoscopy", "endoscopy"])
    ap.add_argument("--inpaint", default="telea", choices=["telea", "ns", "criminisi"])
    ap.add_argument("--no-illum", action="store_true", help="skip illumination correction")
    ap.add_argument("--clahe", action="store_true", help="add CLAHE contrast enhancement")
    ap.add_argument("--max-side", type=int, default=0, help="downscale larger images (speed)")
    ap.add_argument("--demo", action="store_true", help="run on synthetic demo images")
    ap.add_argument("--segment", default="none", choices=["none", "classical", "unet", "medsam", "hybrid"],
                    help="also segment the lesion on the cleaned image (unet/medsam/hybrid need models, see README)")
    ap.add_argument("--unet-weights"); ap.add_argument("--medsam-weights")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="MedSAM box prompt in input-image pixels (default: found automatically)")
    ap.add_argument("--compare-original", action="store_true", help="also segment the uncleaned image")
    args = ap.parse_args()

    cfg = Config(modality=args.modality, inpaint_method=args.inpaint,
                 correct_illumination=not args.no_illum, clahe=args.clahe, max_side=args.max_side,
                 segment=args.segment, unet_weights=args.unet_weights, medsam_weights=args.medsam_weights,
                 seg_compare_original=args.compare_original)

    if args.demo:
        derm, _ = make_demo_dermoscopy(384, seed=3)
        derm = corrupt(derm, seed=3, n_hair=20, ruler=True, ink=True, n_bubbles=2)["corrupted"]
        rng = np.random.default_rng(1)
        endo = add_vignette(make_demo_endoscopy(384, seed=2), 0.4)
        endo, _ = add_bubbles(endo, rng, 4)
        run_one(derm, "demo_dermoscopy", args, dataclasses.replace(cfg, modality="dermoscopy"))
        run_one(endo, "demo_endoscopy", args, dataclasses.replace(cfg, modality="endoscopy"))
        print(f"\nPanels saved in '{args.out}/'")
        return
    if not args.input:
        ap.error("give an input image/folder or use --demo")
    if os.path.isdir(args.input):
        for p in list_images(args.input):
            run_one(p, os.path.splitext(os.path.basename(p))[0], args, cfg)
    else:
        run_one(args.input, os.path.splitext(os.path.basename(args.input))[0], args, cfg)


if __name__ == "__main__":
    main()
