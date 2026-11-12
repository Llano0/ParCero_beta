"""Fill data/external/external.csv from the folder layout of the external photos.

Expected layout (one folder per species):

    data/external/
        ascaris/   *.png|jpg     -> species ascaris,   1 egg per photo
        hookworm/  ...           -> species hookworm,  1 egg per photo
        trichuris/ ...           -> species trichuris, 1 egg per photo
        control/, strongyloides/ -> left out (see IGNORED)
        a-image.png, ...         -> the photos that were already there (left untouched)
        external.csv

The notebooks read the photos from the top level of data/external (not from sub-folders),
so every photo found in a species folder is COPIED to the top level with a short sequential
name (a-n001.png, h-n001.png, t-n001.png). The original name is not used for the
new name, so nothing from the original file name ends up in the data or in the CSV; the
mapping new name -> original path is kept in data/external/_mapping.csv (not versioned).

The script is safe to run again: photos that were already copied are skipped. By default it
only PRINTS what it would do; pass --apply to copy the files and write the CSV (the old
external.csv is saved as external.csv.bak first).

Usage (from the repo root):
    python3 -I data_prep/src/build_external_csv.py              # dry run
    python3 -I data_prep/src/build_external_csv.py --apply
"""

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
COLUMNS = ["image", "n_items", "species", "uso", "notes"]

# folder name (lower case) -> (species written in the CSV, file prefix, eggs per photo)
FOLDERS = {
    "ascaris": ("ascaris", "a", 1),
    "hookworm": ("hookworm", "h", 1),
    "uncinaria": ("hookworm", "h", 1),
    "uncinarias": ("hookworm", "h", 1),
    "trichuris": ("trichuris", "t", 1),
}
# Folders that are left out on purpose:
# - strongyloides: not one of the 3 egg classes (larvae, a later phase of the project, PLAN section 3).
# - control / negatives: slide fields without eggs. The team decided (PLAN section 14, 2026-11-11)
#   that the external evaluation only uses photos with eggs, so they are not copied nor listed.
IGNORED = {"strongyloides", "control", "controles", "negativo", "negativos", "ninguno", "ninguna"}


def read_csv(path):
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/external"))
    ap.add_argument("--uso", default="prueba", help="value of the `uso` column for the new photos")
    ap.add_argument("--apply", action="store_true", help="copy the photos and write the CSV")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist: run this from the repo root")
    csv_path, map_path = root / "external.csv", root / "_mapping.csv"

    old_rows = read_csv(csv_path)
    mapping = read_csv(map_path)
    done_sources = {row["source"] for row in mapping}
    taken = {p.name for p in root.iterdir() if p.is_file()}

    # Continue the numbering of each prefix after the photos that were already copied.
    counters = Counter()
    for row in mapping:
        match = re.match(r"^(\w)-n(\d+)\.", row["new_name"])
        if match:
            counters[match.group(1)] = max(counters[match.group(1)], int(match.group(2)))

    plan, skipped, unknown, ignored = [], Counter(), [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        key = folder.name.strip().lower()
        if key in IGNORED:
            ignored.append(folder.name)
            continue
        if key not in FOLDERS:
            unknown.append(folder.name)
            continue
        species, prefix, n_items = FOLDERS[key]
        for img in sorted(folder.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            source = img.relative_to(root).as_posix()
            if source in done_sources:
                skipped[folder.name] += 1
                continue
            counters[prefix] += 1
            new_name = f"{prefix}-n{counters[prefix]:03d}{img.suffix.lower()}"
            if new_name in taken:
                raise SystemExit(f"{new_name} already exists in {root}: not overwriting")
            taken.add(new_name)
            plan.append({"img": img, "source": source, "new_name": new_name, "species": species,
                         "n_items": n_items, "folder": folder.name})

    print(f"external.csv now has {len(old_rows)} rows")
    per_folder = Counter(item["folder"] for item in plan)
    for folder, n in sorted(per_folder.items()):
        print(f"  new photos from {folder}/: {n}")
    for folder, n in sorted(skipped.items()):
        print(f"  already copied from {folder}/: {n} (skipped)")
    for name in ignored:
        print(f"  [info] {name}/ is left out on purpose (see IGNORED)")
    for name in unknown:
        print(f"  [warn] folder not recognized, ignored: {name}/ "
              f"(known names: {', '.join(sorted(FOLDERS))})")
    if not plan:
        print("nothing new to add")
    for item in plan[:5]:
        print(f"  e.g. {item['source']}  ->  {item['new_name']}  species={item['species'] or '-'}  n_items={item['n_items']}")

    if not args.apply:
        print("\ndry run: nothing was changed. Run again with --apply to copy the photos and write the CSV.")
        return
    if not plan:
        return

    if csv_path.is_file():
        shutil.copy2(csv_path, root / "external.csv.bak")
    for item in plan:
        shutil.copy2(item["img"], root / item["new_name"])
    new_rows = [{"image": item["new_name"], "n_items": item["n_items"], "species": item["species"],
                 "uso": args.uso, "notes": f"folder {item['folder']}"} for item in plan]
    write_csv(csv_path, old_rows + new_rows, COLUMNS)
    write_csv(map_path, mapping + [{"new_name": i["new_name"], "source": i["source"]} for i in plan],
              ["new_name", "source"])

    # Same check the notebooks make: every photo on disk has a row, and every row has a photo.
    on_disk = {p.name for p in root.iterdir() if p.suffix.lower() in IMG_EXTS}
    in_csv = {row["image"] for row in read_csv(csv_path)}
    print(f"\ncopied {len(plan)} photos; external.csv now has {len(in_csv)} rows")
    print("photos without a row:", sorted(on_disk - in_csv) or "none")
    print("rows without a photo:", sorted(in_csv - on_disk) or "none")


if __name__ == "__main__":
    main()
