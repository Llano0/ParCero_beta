#!/usr/bin/env bash
# Download the full Chula-ParasiteEgg-11 export from Roboflow
# (parasites-onxa6/parasites-trgfz v1, YOLOv8 format, CC BY 4.0, ~470 MB zip)
# into its own folder. Needs only bash, curl and python3 (standard library).
#
# Usage (from the repo root; on Apolo, from the login node):
#   export ROBOFLOW_API_KEY=...            # never committed
#   bash data_prep/src/download_chula.sh data/roboflow
#
# Next step: python3 data_prep/src/filter_classes.py --src data/roboflow --dst data/raw
set -euo pipefail

WORKSPACE=parasites-onxa6
PROJECT=parasites-trgfz
VERSION=1
FORMAT=yolov8
DEST=${1:-data/roboflow}

die() { echo "error: $*" >&2; exit 1; }

[[ -f data_prep/src/filter_classes.py ]] || die "run this from the repo root"
[[ -n "${ROBOFLOW_API_KEY:-}" ]] || die "set ROBOFLOW_API_KEY in the environment first"
command -v curl >/dev/null || die "curl not found"
command -v python3 >/dev/null || die "python3 not found"
if [[ -e "$DEST" ]] && [[ -n "$(ls -A "$DEST")" ]]; then
    die "$DEST already exists and is not empty; remove it first to re-download"
fi

# Work in a temp dir next to DEST so a failed or partial download leaves nothing behind.
mkdir -p "$(dirname "$DEST")"
TMP=$(mktemp -d "$(dirname "$DEST")/.chula-download.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

echo "Requesting export link for $WORKSPACE/$PROJECT v$VERSION ($FORMAT)..."
# The key goes through stdin, not argv, so it does not show up in `ps`.
RESPONSE=$(printf '%s' "$ROBOFLOW_API_KEY" | curl -fsS -G --data-urlencode "api_key@-" \
    "https://api.roboflow.com/$WORKSPACE/$PROJECT/$VERSION/$FORMAT")
LINK=$(python3 -c '
import json, sys
data = json.load(sys.stdin)
link = (data.get("export") or {}).get("link")
if not link:
    sys.exit(f"no export link in Roboflow response (keys: {sorted(data)})")
print(link)
' <<<"$RESPONSE")

echo "Downloading (~470 MB)..."
curl -fL --progress-bar -o "$TMP/chula.zip" "$LINK"

echo "Extracting..."
# zipfile.extractall drops absolute paths and '..' components from member names.
python3 -m zipfile -e "$TMP/chula.zip" "$TMP/export"

[[ -f "$TMP/export/data.yaml" ]] || die "export has no data.yaml; unexpected layout"
for split in train valid test; do
    [[ -d "$TMP/export/$split/labels" ]] || die "export has no $split/labels"
done

[[ -d "$DEST" ]] && rmdir "$DEST"
mv "$TMP/export" "$DEST"

echo
grep '^names:' "$DEST/data.yaml"
for split in train valid test; do
    printf '%-6s %6s images\n' "$split" "$(find "$DEST/$split/images" -type f | wc -l)"
done
echo
echo "Done: $DEST"
echo "Next: python3 data_prep/src/filter_classes.py --src $DEST --dst data/raw"
