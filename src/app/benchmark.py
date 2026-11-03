"""Time the full offline pipeline (YOLO + DINOv2 + classifier) on CPU, like the field laptop.

Runs in its own process: the notebook calls it with subprocess because, in this conda
env, torch and scikit-learn ship different OpenMP runtimes that cannot share a process
once scikit-learn's OpenMP code (KMeans, t-SNE) has run.

Usage (from the repo root):
    python src/app/benchmark.py --images data/raw/valid/images --n 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import DEFAULT_BUNDLE, DEFAULT_WEIGHTS, EggPipeline, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if "test" in args.images.resolve().parts:
        raise SystemExit("test is frozen: benchmark on valid or external photos")
    photos = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.n]
    pipeline = EggPipeline(args.weights, args.bundle, device=args.device)
    pipeline.analyze(load_image(photos[0]))  # warm-up
    timings, eggs = [], 0
    for photo in photos:
        image = load_image(photo)
        start = time.perf_counter()
        eggs += len(pipeline.analyze(image))
        timings.append(time.perf_counter() - start)
    print(json.dumps({
        "device": args.device, "photos": len(photos), "eggs_found": eggs,
        "ms_per_photo_mean": round(1000 * sum(timings) / len(timings)),
        "ms_per_photo_max": round(1000 * max(timings)),
        "example": {"photo": photos[0].name, "eggs": pipeline.analyze(load_image(photos[0]))},
    }, indent=2))


if __name__ == "__main__":
    main()
