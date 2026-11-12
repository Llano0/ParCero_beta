"""Crop the circular field of view of microscope photos and paint what is outside it.

Photos taken with a phone through the eyepiece show a bright disc (the field of view) surrounded by
a black ring. This script finds that disc, paints everything outside it with the median colour of
the disc and crops the photo to the disc's bounding box. The orientation is fixed first (EXIF) and
the metadata is dropped when saving.

The original photos are never modified: the result goes to a different folder with the same file
names, and external.csv is copied next to them, so a notebook only needs its EXTERNAL_DIR pointed at
the new folder. Only the photos at the top level of --input are read (not its sub-folders).

If no clear disc is found in a photo (the mask covers almost nothing or almost everything), the
photo is saved unchanged and flagged in the printed table: look at those ones by hand.

Usage (from the repo root):
    python3 -I data_prep/src/crop_field_of_view.py --input data/external --output data/external_fov --preview
    python3 -I data_prep/src/crop_field_of_view.py --input data/external --output data/external_fov --limit 6
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
WORK_SIDE = 600                     # the mask is computed on a copy whose longest side is this
EDGE_TRIM = 0.02                    # the disc border is soft (vignetting): trim this fraction of its radius
MIN_FIELD, MAX_FIELD = 0.10, 0.97   # accepted fraction of the photo covered by the disc
PREVIEW_PHOTOS, PREVIEW_SIDE = 8, 320


def otsu_threshold(gray):
    """Grey level (0-255) that best separates the dark ring from the bright disc."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    levels = np.arange(256)
    below = np.cumsum(hist)
    above = below[-1] - below
    sum_below = np.cumsum(hist * levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_below = sum_below / below
        mean_above = (sum_below[-1] - sum_below) / above
        between = below * above * (mean_below - mean_above) ** 2
    return int(np.nan_to_num(between).argmax())


def field_mask(img):
    """Return (boolean mask of the bright disc at full size, fraction of the photo it covers).

    The mask is None when no clear disc is found."""
    gray = img.convert("L")
    scale = WORK_SIDE / max(gray.size)
    if scale < 1:
        gray = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))))
    values = np.asarray(gray.filter(ImageFilter.GaussianBlur(2)))

    mask = values > otsu_threshold(values)
    mask = ndimage.binary_opening(mask, iterations=2)            # drop small bright specks in the ring
    labels, count = ndimage.label(mask)
    if count == 0:
        return None, 0.0
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    mask = ndimage.binary_fill_holes(labels == 1 + int(np.argmax(sizes)))   # keep the largest region, fill dark debris
    fraction = float(mask.mean())
    if not MIN_FIELD <= fraction <= MAX_FIELD:
        return None, fraction

    radius = np.sqrt(mask.sum() / np.pi)
    mask = ndimage.distance_transform_edt(mask) > EDGE_TRIM * radius   # edges cut by the frame are not trimmed
    full = Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
    return np.asarray(full) > 0, fraction


def process(path):
    """Return (original, result, status) for one photo."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # upright, as the notebooks draw them
    mask, fraction = field_mask(img)
    if mask is None:
        return img, img, f"UNCHANGED (no clear disc, covers {fraction:.0%})"
    pixels = np.asarray(img).copy()
    pixels[~mask] = np.median(pixels[mask][::37], axis=0).astype(np.uint8)   # median colour of the disc
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    result = Image.fromarray(pixels).crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    return img, result, f"ok (disc covers {fraction:.0%})"


def save_preview(pairs, path):
    """Contact sheet: original on the left, result on the right, one photo per row."""
    side = PREVIEW_SIDE
    sheet = Image.new("RGB", (2 * side + 30, len(pairs) * (side + 10) + 10), "white")
    for row, (before, after) in enumerate(pairs):
        for col, image in enumerate((before, after)):
            thumb = image.copy()
            thumb.thumbnail((side, side))
            sheet.paste(thumb, (10 + col * (side + 10), 10 + row * (side + 10)))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="folder with the original photos")
    parser.add_argument("--output", type=Path, required=True, help="new folder for the cropped photos")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N photos (to test)")
    parser.add_argument("--preview", action="store_true",
                        help="also save <output>_preview.png with before/after of the first photos")
    args = parser.parse_args()

    source, target = args.input.resolve(), args.output.resolve()
    if source == target:
        raise SystemExit("--output must be a different folder than --input: the originals are never overwritten")
    photos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.limit:
        photos = photos[: args.limit]
    if not photos:
        raise SystemExit(f"no photos found at the top level of {source}")
    target.mkdir(parents=True, exist_ok=True)

    pairs, unchanged = [], 0
    for photo in photos:
        before, after, status = process(photo)
        options = {"quality": 95} if photo.suffix.lower() in (".jpg", ".jpeg") else {}
        after.save(target / photo.name, **options)
        unchanged += status.startswith("UNCHANGED")
        if len(pairs) < PREVIEW_PHOTOS:
            pairs.append((before, after))
        print(f"{photo.name:<16} {before.width}x{before.height} -> {after.width}x{after.height}  {status}")

    if (source / "external.csv").is_file():
        shutil.copy2(source / "external.csv", target / "external.csv")   # so a notebook only changes EXTERNAL_DIR
    if args.preview:
        save_preview(pairs, target.with_name(target.name + "_preview.png"))   # outside the folder: it holds only photos
    print(f"\n{len(photos)} photos written to {target} ({unchanged} left unchanged)")
    if args.limit:
        print("--limit was used: the folder does not match external.csv, do not point a notebook at it")


if __name__ == "__main__":
    main()
