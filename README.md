# ClearView

Multi-artifact removal and image-quality scoring for **dermatoscopy** and **endoscopy** images,
built on classical image processing (OpenCV, NumPy, SciPy, scikit-image): no deep learning, GPU or special hardware
needed for cleaning and scoring. **Optional:** lesion segmentation on the cleaned image with a **U-Net** (specialist)
and **MedSAM** (generalist), see [Lesion segmentation](#lesion-segmentation-u-net--medsam). This is a
**preprocessing / quality tool, not a diagnostic system**.

## What it does

| Artifact | Modality | Method (module) |
| --- | --- | --- |
| Hair (dark and light) | Dermatoscopy | Oriented morphological black-hat / top-hat bank + local-contrast check + blob removal + thin-component filter (`hair.py`) |
| Ink marks | Dermatoscopy | HSV colour threshold + thin-stroke filter, so blue-grey lesions are not removed (`marks.py`) |
| Ruler ticks | Dermatoscopy | Stroke map, sliding-band column profile, spacing regularity + Fourier peak score (`marks.py`) |
| Air bubbles | Both | Hough circles validated by rim brightness / completeness; only the rim is masked (`marks.py`) |
| Black border, green position inset | Endoscopy | Threshold + largest component + hole fill; HSV green box detector (`fov.py`) |
| Vignetting / uneven light | Both | Homomorphic filtering in the log-luminance domain, optional CLAHE (`illumination.py`) |
| Reconstruction | Both | Telea, Navier-Stokes (OpenCV) and a compact Criminisi exemplar-based inpainter (`inpaint.py`) |
| Quality score | Both | Sharpness, illumination uniformity, contrast, artifact coverage -> 0-100 + Accept / Clean / Re-capture (`quality.py`) |

| Lesion outline (optional) | Both | Classical Otsu baseline, U-Net, MedSAM, or U-Net + MedSAM with agreement / stability checks (`seg/`) |

After cleaning, the detectors are run again on the output (residual artifacts) and the quality score is recomputed, so the system checks its own result.

## Install and run

```bash
pip install -r requirements.txt

python run.py --demo                              # synthetic demo -> samples/
python run.py my_image.jpg --out results          # one image
python run.py my_folder --out results --inpaint ns
python run.py frame.png --modality endoscopy

python serve.py                                   # web app at http://127.0.0.1:8000 (prebuilt UI included)
streamlit run app.py                              # older Streamlit GUI
python -m pytest -q tests                         # unit + API tests
```

Outputs per image: `*_cleaned.png`, `*_mask.png`, `*_panel.png` (input | detected artifacts | cleaned) and a JSON quality report on screen.

## Web app (new UI)

`python serve.py` starts a FastAPI server (`api.py`) that runs the same `clearview.process` pipeline and serves the
React + shadcn/ui front end from `web/dist`. The built UI is included, so Node is **not** needed to run it.
Fonts (Fraunces, Bricolage Grotesque) are bundled, so it works offline.

To change the UI:

```bash
cd web
npm install
npm run dev      # http://localhost:5173, proxies /api to python serve.py on :8000
npm run build    # rebuilds web/dist
```

## Lesion segmentation (U-Net + MedSAM)

The seminar thesis was *"U-Net is the specialist. SAM is the generalist. The field is now combining them."* ClearView
implements that pair as an optional last stage that outlines the lesion **on the cleaned image**:

```
cleaned image -> U-Net (finds the lesion) -> box prompt -> MedSAM (refines the boundary) -> final mask
                                   \______ Dice agreement ______/        \__ prompt stability __/
```

| Method | What it does | Needs |
| --- | --- | --- |
| `classical` | Otsu threshold baseline (dermoscopy only) | nothing |
| `unet` | U-Net mask | PyTorch + `models/unet.pt` |
| `medsam` | MedSAM with a box from you, else from the U-Net, else from the classical mask | PyTorch, `segment-anything`, `models/medsam_vit_b.pth` |
| `hybrid` | U-Net finds the lesion, MedSAM refines it, the two are compared | all of the above |

Anything missing degrades gracefully (hybrid -> medsam or unet -> classical) and the reason is reported in `notes`;
segmentation can never break cleaning. It is **off by default** (`--segment none`).

**What came from the slides, and where it lives**

| Slide component | Code |
| --- | --- |
| U-Net: encoder / bottleneck (1024 ch) / decoder / skip connections, 23 conv layers | `seg/unet.py` `UNet` |
| Elastic deformation (3x3 grid, 10 px std, bicubic) | `seg/common.py` `elastic_deform` |
| Weighted cross-entropy, `w(x) = w_c + w0 exp(-(d1+d2)^2 / 2 sigma^2)`, `w0 = 10`, `sigma = 5` | `seg/common.py` `unet_weight_map`, `seg/unet.py` `weighted_cross_entropy` |
| Overlap-tile inference | `seg/common.py` `predict_tiled` (used when training with `--crop`) |
| Dice loss (came later, V-Net 2016) | `seg/unet.py` `soft_dice_loss`, `--dice-weight` |
| SAM/MedSAM: image encoder runs once, light mask decoder, box prompt only | `seg/medsam.py` (embedding cached; `predict` re-prompts cheaply) |
| SAM ambiguity handling (several masks + IoU score) | MedSAM outputs one mask, so ClearView uses *prompt stability*: jittered boxes, mean pairwise Dice |
| "Specialist vs generalist" comparison | `python -m clearview.evaluate --seg-methods unet,medsam,hybrid` |

Deliberate U-Net deviations: padded convolutions (output = input size, no crop of skip features) and optional BatchNorm.

### Setup

```bash
pip install -r requirements-deep.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
# MedSAM weights (medsam_vit_b.pth) from https://github.com/bowang-lab/MedSAM -> models/medsam_vit_b.pth

python train_unet.py --images ISIC/images --masks ISIC/masks --out models/unet.pt          # GPU recommended
python train_unet.py --demo 150 --epochs 15 --size 128 --base 16                            # synthetic smoke test only
```

Weights are found in this order: command-line flag, `CLEARVIEW_UNET_WEIGHTS` / `CLEARVIEW_MEDSAM_WEIGHTS`, `models/`.
The web API never accepts weight paths from a browser. Only load weights from sources you trust.

### Use

```bash
python run.py img.jpg --segment hybrid                       # adds img_lesion.png + a 4th panel tile
python run.py img.jpg --segment medsam --box 80 60 330 300   # your own box prompt (input-image pixels)
python run.py img.jpg --segment hybrid --compare-original    # how much did cleaning change the outline?
python -m clearview.evaluate --demo 20 --seg-methods unet,medsam,hybrid
```

Web app: pick *Lesion segmentation* in the settings; a **Lesion** tab shows the outlines, the box prompt and the checks.
Unavailable methods are greyed out with the reason. **After pulling this update rebuild the UI once** (`cd web && npm install && npm run build`);
the prebuilt `web/dist` predates the segmentation controls. API: `options.segment`, `options.seg_compare_original`,
`options.box` (normalised x0,y0,x1,y1), `GET /api/segmentation/status`.

### Review flags

A result is marked *worth a look* when the lesion touches the frame edge, almost nothing was segmented, U-Net and MedSAM
disagree (Dice < 0.80) or MedSAM's mask changes with small box shifts (stability < 0.90). These thresholds are
hand-picked starting values.

### Caveats for segmentation

* The deep-learning code paths (U-Net, MedSAM) could not be run where this update was written (no PyTorch, no weights, no network).
  They follow the published architectures and the official MedSAM inference recipe, and everything around them is tested
  with stand-in models. **Run `pytest tests/test_seg.py` with PyTorch installed (it adds U-Net shape/training tests), then try
  `--segment hybrid` on a few real images before trusting any numbers.**
* MedSAM was trained mainly on CT, MRI and endoscopy; dermoscopy is not its strong suit, and vessel-like or branching structures are hard for a box prompt.
* A U-Net trained on a few images, or only on the synthetic demo images, will not generalise. Measure it with `evaluate` on real data.
* The classical baseline assumes a dark lesion on lighter skin and is skipped for endoscopy.
* CPU inference: the MedSAM image encoder takes seconds per image. Results are cached per cleaned image, but the web UI re-runs on every setting change, so keep segmentation off while tuning other settings.
* Research models, not medical devices.

## Evaluation

Real artifact-free "before/after" pairs do not exist, so evaluation injects **known synthetic artifacts**
(random curved hair, ruler, ink loop, bubbles, vignetting) into clean images and compares with the artifact-free reference.

```bash
python -m clearview.evaluate --demo 20 --compare-inpainting --with-criminisi      # built-in synthetic images
python -m clearview.evaluate --images ISIC/images --masks ISIC/masks --n 100        # real dataset (ISIC 2018 Task 1 layout)
```

With real images, the downstream test runs a fixed Otsu lesion segmentation before and after cleaning and reports the IoU change.

Results on 20 built-in synthetic dermatoscopy images (384 px, heavy artifacts, about 27% of the image covered on average):

| | Telea | Navier-Stokes | Criminisi |
| --- | --- | --- | --- |
| PSNR (corrupted -> restored) | 19.3 -> 26.0 dB | 19.3 -> 26.0 dB | 19.3 -> 23.5 dB (earlier run) |
| SSIM | 0.80 -> 0.91 | 0.80 -> 0.91 | 0.80 -> 0.88 (earlier run) |
| Lesion-segmentation IoU (clean / corrupted / restored) | 0.99 / 0.81 / 0.93 | 0.99 / 0.81 / 0.94 | 0.99 / 0.81 / 0.92 |
| Quality score | 57.6 -> 80.6 | 57.6 -> 80.2 | 57.6 -> 75.2 |
| Time per image | 0.1 s | 0.1 s | about 6 s |

Illumination non-uniformity drops from 0.115 to 0.066. Artifact mask Dice is about 0.63 (the mask is slightly wider than the ground truth on purpose, to cover soft edges).
Exact numbers vary a little between runs of the code as you tune it; rerun the command to reproduce them.

## Honest limitations

* Evaluated on **synthetic** artifacts and synthetic demo skin/endoscopy images. Real-data performance must be measured by running `evaluate` on ISIC / HAM10000 / PH2 / Kvasir / CVC-ClinicDB; expect lower numbers on real, messier images.
* Ruler detection handles **axis-aligned** rulers only. Diagonal rulers are not detected.
* Hair that lies exactly on the lesion border or is wider than about 0.6 x the kernel length (about 3% of the image side) is not detected; strongly curled or very light hair may be missed.
* Dermatoscopy vs endoscopy "auto" mode is a heuristic (it looks for the green position inset). Pass `--modality` explicitly when you know it.
* Classical inpainting blurs texture on large masks; if more than 45% of the valid area is flagged, the image is **not** inpainted and is marked Re-capture-worthy.
* Criminisi inpainting is educational/compact and slow; use Telea or NS for real use.
* Lesion-segmentation checks (agreement, stability, edge contact) are heuristics, not validated.
* Quality-score weights and thresholds in `quality.py` are hand-tuned starting values, not clinically validated.

## Project layout

```
clearview/
  fov.py           valid region, border and green-inset detection
  hair.py          hair detector
  marks.py         ink, ruler, bubble detectors
  illumination.py  homomorphic filter, CLAHE, non-uniformity measure
  inpaint.py       Telea / NS / Criminisi
  quality.py       image quality score
  pipeline.py      Config, process()
  synthetic.py     demo images + artifact injection (ground truth)
  evaluate.py      PSNR / SSIM / Dice / downstream-segmentation evaluation
  seg/             lesion segmentation: classical, unet.py, medsam.py, engine.py (hybrid), common.py (helpers)
run.py             command-line tool
train_unet.py      train the U-Net (needs PyTorch)
models/            put unet.pt and medsam_vit_b.pth here (not bundled)
requirements-deep.txt  optional: PyTorch (+ segment-anything)
api.py             FastAPI backend + static UI server
serve.py           starts the web app
web/               React + Tailwind + shadcn/ui source (web/dist = prebuilt)
app.py             older Streamlit GUI (theme in .streamlit/config.toml)
tests/             pytest unit tests
samples/           example outputs from `python run.py --demo`
```

## Datasets (public, use with attribution)

ISIC archive / ISIC 2018 Task 1, HAM10000, PH2 (dermatoscopy); Kvasir, HyperKvasir, CVC-ClinicDB (endoscopy).
The project does not bundle any patient data.
