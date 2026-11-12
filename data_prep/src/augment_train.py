"""Build a degraded copy of the TRAIN split only, to test whether quality and scene
augmentations help the detector on phone photos taken through the eyepiece.

Ultralytics has no CLI option for blur, noise, JPEG, circular vignette or an eyepiece
pointer, and its online transforms run after mosaic (they could black out a different
egg of the mosaic). Doing it offline lets us keep every box visible and keeps the
labels untouched: none of these transforms moves pixels, so the boxes stay valid.

Each train photo is replaced, with probability --p, by a degraded version (the other
photos are copied as they are), so the augmented set has exactly the same number of
photos as the original one and the comparison is fair. valid is NOT touched: the data
yaml points to data/raw/valid, and test is not listed at all.

Transforms (each with its own probability; at least one is applied):
    pointer   tapered dark or red needle that points at a random egg without touching it
              (in Chula only ascaris photos carry a red pointer: this spreads it over all classes)
    vignette  circular field with black outside, always keeping every box inside
    blur      Gaussian blur (out of focus)
    noise     Gaussian sensor noise
    jpeg      JPEG recompression at low quality

Usage:
    python3 data_prep/src/augment_train.py --raw data/raw --out data/aug --p 0.8
Then train with:  data=data/aug/data.yaml
"""

import argparse
import csv
import io
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
OP_PROBS = {"pointer": 0.30, "vignette": 0.50, "blur": 0.50, "noise": 0.40, "jpeg": 0.50}


def read_boxes(label_file, W, H):
    """[(x1, y1, x2, y2)] in pixels. Polygons become their bounding box."""
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            cx, cy, w, h = vals
            out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        else:
            xs, ys = vals[0::2], vals[1::2]
            out.append((min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H))
    return out


def inside_any(x, y, boxes, pad=0.0):
    return any(x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad for x1, y1, x2, y2 in boxes)


def add_pointer(img, boxes, rng):
    """Draw a tapered needle whose tip stops outside the box it points at."""
    W, H = img.size
    for _ in range(10):  # retry until the needle touches no box
        x1, y1, x2, y2 = rng.choice(boxes)
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        ang = rng.uniform(0, 2 * math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        tip_d = side * rng.uniform(0.8, 1.4)  # 0.8 * side > half-diagonal / side (0.707)
        length = W * rng.uniform(0.35, 0.7)
        tip = (cx + dx * tip_d, cy + dy * tip_d)
        tail = (tip[0] + dx * length, tip[1] + dy * length)
        if any(inside_any(tip[0] + dx * length * t / 10, tip[1] + dy * length * t / 10, boxes, pad=4)
               for t in range(11)):
            continue
        nx, ny = -dy, dx
        half_tail, half_tip = rng.uniform(3, 7), rng.uniform(0.8, 1.5)
        color = rng.choice([(20, 20, 25), (190, 30, 30)])
        ImageDraw.Draw(img).polygon([
            (tip[0] + nx * half_tip, tip[1] + ny * half_tip),
            (tip[0] - nx * half_tip, tip[1] - ny * half_tip),
            (tail[0] - nx * half_tail, tail[1] - ny * half_tail),
            (tail[0] + nx * half_tail, tail[1] + ny * half_tail),
        ], fill=color)
        return img, True
    return img, False


def add_vignette(img, boxes, rng):
    """Black outside a circle that keeps every box (with 5 % slack) inside."""
    W, H = img.size
    cx = W / 2 + rng.uniform(-0.08, 0.08) * W
    cy = H / 2 + rng.uniform(-0.08, 0.08) * H
    needed = max(math.hypot(x - cx, y - cy) for x1, y1, x2, y2 in boxes
                 for x in (x1, x2) for y in (y1, y2)) * 1.05 + 8
    radius = needed + rng.uniform(0, 0.2) * W
    if needed >= 0.75 * W:  # the circle would not black anything out
        return img, False
    yy, xx = np.mgrid[0:H, 0:W]
    outside = (xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2
    arr = np.asarray(img).copy()
    arr[outside] = [rng.randint(0, 25), rng.randint(0, 25), rng.randint(0, 30)]
    return Image.fromarray(arr), True


def add_blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.5)))


def add_noise(img, rng, np_rng):
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np_rng.normal(0, rng.uniform(3, 14), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def augment(img, boxes, rng, np_rng):
    """Returns (image, [applied ops], jpeg quality to save with)."""
    while True:
        chosen = [op for op, p in OP_PROBS.items() if rng.random() < p]
        if chosen:
            break
    applied = []
    if "pointer" in chosen and boxes:
        img, ok = add_pointer(img, boxes, rng)
        applied += ["pointer"] * ok
    if "vignette" in chosen and boxes:
        img, ok = add_vignette(img, boxes, rng)
        applied += ["vignette"] * ok
    if "blur" in chosen:
        img, applied = add_blur(img, rng), applied + ["blur"]
    if "noise" in chosen:
        img, applied = add_noise(img, rng, np_rng), applied + ["noise"]
    quality = 95
    if "jpeg" in chosen:
        quality = rng.randint(25, 75)
        applied.append("jpeg")
    return img, applied, quality


def names_lines(raw_yaml):
    """The `nc:` and `names:` lines of the original data.yaml, untouched."""
    return [l for l in raw_yaml.read_text().splitlines() if l.startswith(("nc:", "names:"))]


def make_preview(pairs, path, thumb=320):
    """Top row: original; bottom row: augmented. For checking by eye."""
    canvas = Image.new("RGB", (thumb * len(pairs), thumb * 2), "white")
    for i, (orig, aug) in enumerate(pairs):
        canvas.paste(orig.resize((thumb, thumb)), (i * thumb, 0))
        canvas.paste(aug.resize((thumb, thumb)), (i * thumb, thumb))
    canvas.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/aug"))
    ap.add_argument("--p", type=float, default=0.8, help="probability of degrading each train photo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", type=int, default=6, help="photos shown in preview.jpg (0 = none)")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: remove it or pick another --out")
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)

    img_out, lbl_out = args.out / "train" / "images", args.out / "train" / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    manifest, pairs = [], []
    src_images = sorted(p for p in (args.raw / "train" / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_path in src_images:
        lbl_path = args.raw / "train" / "labels" / f"{img_path.stem}.txt"
        shutil.copy(lbl_path, lbl_out / lbl_path.name)  # labels never change
        dst = img_out / img_path.name
        applied = []
        if rng.random() < args.p:
            img = Image.open(img_path).convert("RGB")
            aug, applied, quality = augment(img.copy(), read_boxes(lbl_path, *img.size), rng, np_rng)
            if applied:
                aug.save(dst, quality=quality)
                if len(pairs) < args.preview:
                    pairs.append((img, aug))
        if not applied:
            shutil.copy(img_path, dst)
        manifest.append({"image": img_path.name, "ops": "+".join(applied)})

    with open(args.out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ops"])
        writer.writeheader()
        writer.writerows(manifest)

    # train = degraded copy; val = the untouched valid split; test is not listed on purpose
    (args.out / "data.yaml").write_text("\n".join([
        f"path: {args.out.resolve()}",
        "train: train/images",
        f"val: {(args.raw / 'valid' / 'images').resolve()}",
        "",
        *names_lines(args.raw / "data.yaml"),
        "",
    ]))
    if pairs:
        make_preview(pairs, args.out / "preview.jpg")

    changed = [m for m in manifest if m["ops"]]
    print(f"{len(manifest)} train photos, {len(changed)} degraded ({len(changed) / len(manifest):.0%})")
    for op in OP_PROBS:
        print(f"  {op:9s}{sum(op in m['ops'].split('+') for m in manifest):5d}")
    print(f"Wrote {args.out}/data.yaml, manifest.csv" + (", preview.jpg" if pairs else ""))


if __name__ == "__main__":
    main()
