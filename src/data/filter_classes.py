"""Keep only Ascaris / Trichuris / Hookworm from the Roboflow YOLOv8 export and
build a fixed, stratified train/valid/test split. Standard library only, so it
runs on any server with python3.

Labels are remapped to 0=ascaris, 1=hookworm, 2=trichuris. The split is done per
source image (all eggs of an image stay together) and is deterministic for a
given --seed, so the CNN and the embedding pipeline see the same test set.

Each split also gets a COCO file with the same boxes, for COCO-based loaders:
    <dst>/<split>/_annotations.coco.json   # file_name = "images/<name>"
COCO category ids start at 1: 1=ascaris, 2=hookworm, 3=trichuris.

Usage:
    python3 src/data/filter_classes.py --src data/roboflow --dst data/raw
"""

import argparse
import json
import random
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path

# Keyword in the original class name -> (new index, label)
KEEP = {"ascaris": (0, "ascaris"), "hookworm": (1, "hookworm"), "trichuris": (2, "trichuris")}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_names(data_yaml):
    """Parse the `names: [...]` line of a Roboflow data.yaml without PyYAML."""
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            raw = line.split(":", 1)[1].strip().strip("[]")
            return [n.strip().strip("'\"") for n in raw.split(",")]
    raise ValueError(f"no 'names:' line in {data_yaml}")


def image_size(path):
    """(width, height) read from the PNG/JPEG header, without PIL."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head.startswith(b"\x89PNG"):
            return struct.unpack(">II", head[16:24])
        f.seek(2)
        while True:
            marker, length = struct.unpack(">2sH", f.read(4))
            if marker[1] in (0xC0, 0xC1, 0xC2):  # start of frame: precision, h, w
                h, w = struct.unpack(">xHH", f.read(5))
                return w, h
            f.seek(length - 2, 1)


def to_coco(images, labels):
    """images: [(dst_image_path, yolo_lines)] of one split -> COCO dict."""
    coco = {
        "images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": "egg"} for i, name in enumerate(labels)],
    }
    for img_id, (img_path, lines) in enumerate(images, start=1):
        W, H = image_size(img_path)
        coco["images"].append({"id": img_id, "file_name": f"images/{img_path.name}", "width": W, "height": H})
        for line in lines:
            cls, *vals = line.split()
            vals = list(map(float, vals))
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1, y1 = cx - w / 2, cy - h / 2
            else:  # polygon -> its bbox
                xs, ys = vals[0::2], vals[1::2]
                x1, y1, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            box = [round(x1 * W, 2), round(y1 * H, 2), round(w * W, 2), round(h * H, 2)]
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": int(cls) + 1,
                "bbox": box, "area": round(box[2] * box[3], 2), "iscrowd": 0,
            })
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="unzipped Roboflow export")
    ap.add_argument("--dst", type=Path, default=Path("data/raw"))
    ap.add_argument("--ratios", default="0.7,0.15,0.15", help="train,valid,test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = read_names(args.src / "data.yaml")
    remap = {}
    for old_idx, name in enumerate(names):
        for key, (new_idx, _) in KEEP.items():
            if key in name.lower():
                remap[old_idx] = new_idx
    print("Kept:", {names[o]: n for o, n in remap.items()})

    # image -> (label lines remapped, class of the image)
    by_class = defaultdict(list)
    for lbl in sorted(args.src.glob("*/labels/*.txt")):
        lines, classes = [], set()
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) in remap:
                new = remap[int(parts[0])]
                lines.append(" ".join([str(new)] + parts[1:]))
                classes.add(new)
        if not lines:
            continue
        img = next((p for p in lbl.parent.parent.joinpath("images").glob(lbl.stem + ".*")
                    if p.suffix.lower() in IMG_EXTS), None)
        if img is None:
            continue
        # Images never mix classes in this dataset; min() just picks a stratum if they did.
        by_class[min(classes)].append((img, lines))

    ratios = [float(r) for r in args.ratios.split(",")]
    rng = random.Random(args.seed)
    counts = Counter()
    written = defaultdict(list)  # split -> [(dst_image_path, lines)]
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        n_train = round(len(items) * ratios[0])
        n_valid = round(len(items) * ratios[1])
        parts = {"train": items[:n_train], "valid": items[n_train:n_train + n_valid],
                 "test": items[n_train + n_valid:]}
        for split, chunk in parts.items():
            (args.dst / split / "images").mkdir(parents=True, exist_ok=True)
            (args.dst / split / "labels").mkdir(parents=True, exist_ok=True)
            for img, lines in chunk:
                shutil.copy2(img, args.dst / split / "images" / img.name)
                (args.dst / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
                written[split].append((args.dst / split / "images" / img.name, lines))
                counts[(split, cls)] += 1

    labels = [lbl for _, (_, lbl) in sorted(KEEP.items(), key=lambda kv: kv[1][0])]
    for split, images in written.items():
        (args.dst / split / "_annotations.coco.json").write_text(json.dumps(to_coco(images, labels)))
    (args.dst / "data.yaml").write_text(
        f"path: {args.dst.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(labels)}\nnames: {labels}\n"
    )

    print("\nimages per split".ljust(18) + "".join(l.rjust(11) for l in labels))
    for split in ("train", "valid", "test"):
        print(split.ljust(17) + "".join(str(counts[(split, i)]).rjust(11) for i in range(len(labels))))
    print(f"\nWrote {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
