"""Crop every egg from a Roboflow YOLO export into its own image.

Input layout (Roboflow "YOLOv8" export):
    data/raw/data.yaml
    data/raw/{train,valid,test}/images/*.jpg
    data/raw/{train,valid,test}/labels/*.txt   # cls cx cy w h (normalized)

Output:
    data/crops/<split>/<label>/<image_stem>_<i>.png
    data/crops/crops.csv   # one row per crop, used by every later step

Usage:
    python src/data/crop_eggs.py --raw data/raw --out data/crops --margin 0.15
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

# Keyword in the dataset class name -> label we use. Anything else is dropped.
TARGET_CLASSES = {
    "ascaris": "ascaris",
    "trichuris": "trichuris",
    "hookworm": "hookworm",
}
SPLITS = ("train", "valid", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def map_class_names(names):
    """Map YOLO class index -> our label, keeping only the 3 target classes."""
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    mapping = {}
    for idx, name in enumerate(names):
        for key, label in TARGET_CLASSES.items():
            if key in name.lower():
                mapping[idx] = label
    return names, mapping


def source_group(stem):
    """Roboflow renames files to '<original>_jpg.rf.<hash>'; augmented copies share
    the original part. Crops are grouped by it so they never straddle splits."""
    return stem.split(".rf.")[0]


def read_boxes(label_file, img_w, img_h):
    """Yield (class_idx, x1, y1, x2, y2) in pixels. Polygons become their bbox."""
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            cx, cy, w, h = vals
            x1, x2 = (cx - w / 2) * img_w, (cx + w / 2) * img_w
            y1, y2 = (cy - h / 2) * img_h, (cy + h / 2) * img_h
        else:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2 = min(xs) * img_w, max(xs) * img_w
            y1, y2 = min(ys) * img_h, max(ys) * img_h
        yield cls, x1, y1, x2, y2


def expand(x1, y1, x2, y2, margin, img_w, img_h):
    """Grow the box by `margin` of its size on each side, clamped to the image."""
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, int(round(x1 - dw))),
        max(0, int(round(y1 - dh))),
        min(img_w, int(round(x2 + dw))),
        min(img_h, int(round(y2 + dh))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/crops"))
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--min-size", type=int, default=16, help="skip boxes smaller than this (px)")
    args = ap.parse_args()

    names, class_map = map_class_names(yaml.safe_load((args.raw / "data.yaml").read_text())["names"])
    print("Dataset classes:", names)
    print("Kept:", {names[i]: lbl for i, lbl in class_map.items()})

    rows, skipped = [], Counter()
    for split in SPLITS:
        img_dir, lbl_dir = args.raw / split / "images", args.raw / split / "labels"
        if not img_dir.exists():
            print(f"[warn] missing split: {split}")
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                skipped["no_label_file"] += 1
                continue
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            for i, (cls, *box) in enumerate(read_boxes(lbl_path, W, H)):
                label = class_map.get(cls)
                if label is None:
                    skipped["other_class"] += 1
                    continue
                x1, y1, x2, y2 = expand(*box, args.margin, W, H)
                if min(x2 - x1, y2 - y1) < args.min_size:
                    skipped["too_small"] += 1
                    continue
                crop_path = args.out / split / label / f"{img_path.stem}_{i}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                img.crop((x1, y1, x2, y2)).save(crop_path)
                rows.append({
                    "crop_path": str(crop_path),
                    "label": label,
                    "split": split,
                    "source_image": str(img_path),
                    "group": source_group(img_path.stem),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    # original box size (no margin) — egg size is a useful feature on its own
                    "box_w": round(box[2] - box[0], 1),
                    "box_h": round(box[3] - box[1], 1),
                })

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "crops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary ---------------------------------------------------------
    print(f"\n{len(rows)} crops written to {args.out}; skipped: {dict(skipped)}")
    counts = Counter((r["split"], r["label"]) for r in rows)
    labels = sorted({r["label"] for r in rows})
    print("\n" + "split".ljust(8) + "".join(l.rjust(12) for l in labels))
    for split in SPLITS:
        print(split.ljust(8) + "".join(str(counts[(split, l)]).rjust(12) for l in labels))

    per_group = defaultdict(set)
    imgs_per_group = defaultdict(set)
    for r in rows:
        per_group[r["group"]].add(r["split"])
        imgs_per_group[r["group"]].add(r["source_image"])
    leaks = {g: s for g, s in per_group.items() if len(s) > 1}
    augmented = sum(len(v) > 1 for v in imgs_per_group.values())
    print(f"\nSource images: {len(per_group)} originals, {augmented} with augmented copies")
    print(f"Originals appearing in more than one split (leakage): {len(leaks)}")
    for g, s in list(leaks.items())[:10]:
        print(f"  {g}: {sorted(s)}")


if __name__ == "__main__":
    main()
