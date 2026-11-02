"""Run the YOLO detector on photos and cut one crop per detected egg, in the same
format as crop_eggs.py, so the ML branch (extract_embeddings.py) and the DL branch
can classify what the detector found instead of the hand-drawn boxes.

Writes <out>/crops/<photo>_<i>.png and <out>/crops.csv with the columns that
extract_embeddings.py needs (crop_path, label, split, group) plus the detector's
score. When --raw is given and the photos belong to a split of data/raw, every
detection is matched to the ground-truth boxes (IoU >= --match-iou): `label` is the
species of the matched box, or "none" for a detection that matches no egg (false
positive). Without --raw (external photos) `label` is "unlabeled".

Also prints the detector's recall and precision at the chosen --conf, and how many
eggs it missed. The test split is frozen: it is refused unless --final is passed.

Usage (from the repo root):
    python3 src/detection/detect_crops.py \
        --weights artifacts/yolo/h_single_aug/weights/best.pt \
        --images data/raw/valid/images --raw data/raw --split valid \
        --out artifacts/detected/valid --conf 0.10
    python3 src/detection/detect_crops.py \
        --weights artifacts/yolo/h_single_aug/weights/best.pt \
        --images data/external --split external --out artifacts/detected/external
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from crop_eggs import IMG_EXTS, expand, read_boxes, source_group  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def list_images(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="a folder or a single image")
    ap.add_argument("--split", default="external", help="value written in the `split` column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=None,
                    help="data/raw, to match detections with the ground-truth boxes")
    ap.add_argument("--conf", type=float, default=0.10, help="low on purpose: the classifier filters")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--margin", type=float, default=0.15, help="same as crop_eggs.py")
    ap.add_argument("--min-size", type=int, default=16)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--final", action="store_true", help="allow the frozen test split")
    args = ap.parse_args()

    if "test" in args.images.resolve().parts and not args.final:
        raise SystemExit("test is frozen: use it only for the final evaluation (--final)")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: remove it or pick another --out")

    class_names = None
    if args.raw:
        names = yaml.safe_load((args.raw / "data.yaml").read_text())["names"]
        class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"no images found in {args.images}")
    model = YOLO(str(args.weights))
    crop_dir = args.out / "crops"
    crop_dir.mkdir(parents=True)

    rows, stats = [], Counter()
    for r in model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf,
                           device=args.device, stream=True, verbose=False):
        img_path = Path(r.path)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        dets = sorted(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()), key=lambda d: -d[1])

        gt = []  # [(label, box)] of this photo
        if args.raw:
            lbl = args.images / ".." / "labels" / f"{img_path.stem}.txt"
            if lbl.exists():
                gt = [(class_names[c], tuple(b)) for c, *b in read_boxes(lbl, W, H)]
        used = set()
        stats["photos"] += 1
        stats["gt_eggs"] += len(gt)

        for i, (box, score) in enumerate(dets):
            label, best_iou = ("unlabeled" if not args.raw else "none"), 0.0
            cands = [(iou(box, b), j) for j, (_, b) in enumerate(gt) if j not in used]
            if cands and max(cands)[0] >= args.match_iou:
                best_iou, j = max(cands)
                used.add(j)
                label = gt[j][0]
            stats["detections"] += 1
            stats["matched"] += label not in ("none", "unlabeled")

            x1, y1, x2, y2 = expand(*box, args.margin, W, H)
            if min(x2 - x1, y2 - y1) < args.min_size:
                stats["too_small"] += 1
                continue
            crop_path = crop_dir / f"{img_path.stem}_{i}.png"
            img.crop((x1, y1, x2, y2)).save(crop_path)
            rows.append({
                "crop_path": str(crop_path), "label": label, "split": args.split,
                "source_image": str(img_path), "group": source_group(img_path.stem),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "box_w": round(box[2] - box[0], 1), "box_h": round(box[3] - box[1], 1),
                "det_conf": round(score, 4), "match_iou": round(best_iou, 3),
            })

    if not rows:
        raise SystemExit("no detections: try a lower --conf or check the weights")
    with open(args.out / "crops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{stats['photos']} photos, {stats['detections']} detections at conf >= {args.conf}, "
          f"{len(rows)} crops written to {args.out}/crops ({stats['too_small']} too small)")
    if args.raw and stats["gt_eggs"]:
        recall = stats["matched"] / stats["gt_eggs"]
        precision = stats["matched"] / max(stats["detections"], 1)
        print(f"Against the ground truth (IoU >= {args.match_iou}): "
              f"recall {recall:.3f}  precision {precision:.3f}  "
              f"missed eggs {stats['gt_eggs'] - stats['matched']}  "
              f"false positives {stats['detections'] - stats['matched']}")
    print("Labels:", dict(Counter(r["label"] for r in rows)))


if __name__ == "__main__":
    main()
