"""Train the ClearView U-Net (the "specialist").

    python train_unet.py --images ISIC/images --masks ISIC/masks --out models/unet.pt
    python train_unet.py --demo 150 --epochs 15 --size 128 --base 16     # quick smoke test on synthetic images

Recipe from the U-Net paper: elastic-deformation augmentation, weighted cross-entropy with the border weight
map, small data. The paper used SGD with momentum 0.99 and batch 1 (`--optim sgd --lr 0.01 --batch 1`);
Adam is the default here. Needs PyTorch (pip install -r requirements-deep.txt).

Weights trained with --demo only know the synthetic skin of this project. They are a pipeline test, not a model.
"""
import argparse
from types import SimpleNamespace

from clearview.evaluate import load_dataset
from clearview.synthetic import make_demo_dermoscopy


def main():
    ap = argparse.ArgumentParser(description="Train the ClearView U-Net")
    ap.add_argument("--images"); ap.add_argument("--masks")
    ap.add_argument("--demo", type=int, default=0, help="train on N synthetic images (smoke test only)")
    ap.add_argument("--n", type=int, default=2000, help="max real images to load")
    ap.add_argument("--out", default="models/unet.pt")
    ap.add_argument("--size", type=int, default=256, help="images are resized to size x size")
    ap.add_argument("--crop", type=int, default=0, help="train on random crops of this size, infer with overlap-tile")
    ap.add_argument("--base", type=int, default=64, help="first-level channels (paper: 64; use 16-32 on a CPU)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--no-bn", action="store_true", help="plain convs without BatchNorm, as in the paper")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--w0", type=float, default=10.0); ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--dice-weight", type=float, default=0.0, help="add a soft-Dice term (post-U-Net; V-Net 2016)")
    ap.add_argument("--val-frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not a.demo and not (a.images and a.masks):
        ap.error("give --images DIR --masks DIR, or --demo N")

    try:
        from clearview.seg.unet import fit
    except ImportError as e:
        raise SystemExit(f"PyTorch is required: pip install -r requirements-deep.txt ({e})")

    if a.demo:
        pairs = [make_demo_dermoscopy(max(a.size, 192), seed=5000 + i) for i in range(a.demo)]     # seeds != evaluate's
        images, masks = [p[0] for p in pairs], [p[1] for p in pairs]
        print("Training on synthetic images: the result is a smoke test, not a usable model.")
    else:
        items = load_dataset(SimpleNamespace(demo=0, images=a.images, masks=a.masks, n=a.n, size=max(a.size, 384)))
        items = [(n, i, m) for n, i, m in items if m is not None]
        if not items:
            raise SystemExit("No image/mask pairs found. Mask names: <stem>.png, <stem>_segmentation.png, <stem>_mask.png")
        images, masks = [i for _, i, _ in items], [m for _, _, m in items]
        print(f"{len(items)} image/mask pairs")
    r = fit(images, masks, a.out, size=a.size, crop=a.crop, base=a.base, depth=a.depth, batch_norm=not a.no_bn,
            epochs=a.epochs, batch=a.batch, lr=a.lr, optim=a.optim, w0=a.w0, sigma=a.sigma,
            dice_weight=a.dice_weight, val_frac=a.val_frac, seed=a.seed, device=a.device)
    print(f"best validation Dice: {r['best']:.3f}" if r["history"][-1]["val_dice"] is not None else "done")


if __name__ == "__main__":
    main()
