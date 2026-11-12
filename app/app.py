"""Illustrative local demo of the egg pipeline. It is a support tool, never a diagnosis.

photo -> YOLO 1-class detector -> crop with 15 % margin -> EfficientNet-B0 (3 seeds averaged)
-> species + confidence for every egg. The page (app/static/index.html) keeps the per-sample log
in the browser: there is no database and nothing is written to disk.

Only the Python standard library is used for the server, so it runs in the `geohelmintos` env
as is. Run from the repo root (the weights in dl/outputs/ are not versioned):

    python app/app.py                 # then open http://127.0.0.1:7860
    python app/app.py --port 7861     # if the port is taken
"""

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import models as tv_models
from torchvision import transforms as T
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DETECTOR = ROOT / "dl/outputs/yolo/yolo_1class_aug/weights/best.pt"
CLASSIFIERS = [ROOT / f"dl/outputs/transfer_efficientnet_b0/seed{s}/best.pt" for s in (42, 43, 44)]
CLASSES = ["ascaris", "hookworm", "trichuris"]
LABELS = {"ascaris": "Ascaris", "hookworm": "Uncinaria", "trichuris": "Trichuris"}

DET_CONF = 0.25          # same detector threshold as the notebooks
MARGIN = 0.15            # same crop margin as data_prep/src/crop_eggs.py
MIN_SIDE = 16
MAX_BOXES = 50
REVIEW_BELOW = 0.65      # below this classifier confidence the egg is flagged for a professional
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_UPLOAD = 40 * 1024 * 1024
PREVIEW_SIDE = 1600      # longest side of the annotated image sent back to the page

TF = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(MEAN, STD)])


class TransferNet(nn.Module):
    # Same layout as dl/notebooks/transfer_learning.ipynb: pretrained features + the CNN base head.
    def __init__(self, features, channels, n_classes=3):
        super().__init__()
        self.features = features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(channels, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def load_models():
    missing = [str(p.relative_to(ROOT)) for p in [DETECTOR, *CLASSIFIERS] if not p.is_file()]
    if missing:
        raise SystemExit("missing weights (dl/outputs/ is not versioned):\n  " + "\n  ".join(missing))
    detector = YOLO(str(DETECTOR))
    classifiers = []
    for path in CLASSIFIERS:
        model = TransferNet(tv_models.efficientnet_b0(weights=None).features, 1280)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        classifiers.append(model.eval())
    return detector, classifiers


def pad_to_square(img):
    # Pad with the crop's mean colour instead of stretching, as in training.
    side = max(img.size)
    fill = tuple(int(c) for c in np.asarray(img).reshape(-1, 3).mean(axis=0))
    return ImageOps.pad(img, (side, side), color=fill, centering=(0.5, 0.5))


def expand(box, width, height):
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return (max(0, int(round(x1 - dw))), max(0, int(round(y1 - dh))),
            min(width, int(round(x2 + dw))), min(height, int(round(y2 + dh))))


def draw(img, eggs):
    scale = min(1.0, PREVIEW_SIDE / max(img.size))
    out = img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img.copy()
    d = ImageDraw.Draw(out)
    line = max(3, round(max(out.size) / 260))
    font = ImageFont.load_default(size=max(14, round(max(out.size) / 55)))
    for egg in eggs:
        x1, y1, x2, y2 = (v * scale for v in egg["box"])
        color = (217, 119, 6) if egg["review"] else (22, 163, 74)
        d.rectangle((x1, y1, x2, y2), outline=color, width=line)
        text = f'{egg["n"]}. {LABELS[egg["species"]]} {egg["confidence"]:.0%}'
        tx, ty = x1, max(0, y1 - font.size - 2 * line)
        bbox = d.textbbox((tx, ty), text, font=font)
        d.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=color)
        d.text((tx, ty), text, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Pipeline:
    def __init__(self):
        self.detector, self.classifiers = load_models()
        self.lock = threading.Lock()   # one photo at a time

    def analyze(self, data):
        # Upright first: phone JPGs carry EXIF rotation and YOLO boxes must match the crops.
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        width, height = img.size
        with self.lock, torch.no_grad():
            result = self.detector.predict(source=img, imgsz=640, conf=DET_CONF, max_det=MAX_BOXES,
                                           device="cpu", verbose=False)[0]
            found = sorted(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist()), reverse=True)
            crops, kept = [], []
            for score, box in found:
                x1, y1, x2, y2 = expand(box, width, height)
                if min(x2 - x1, y2 - y1) < MIN_SIDE:
                    continue
                crops.append(pad_to_square(img.crop((x1, y1, x2, y2))))
                kept.append((score, box))
            probs = np.zeros((0, len(CLASSES)))
            if crops:
                x = torch.stack([TF(c) for c in crops])
                probs = torch.stack([F.softmax(m(x), 1) for m in self.classifiers]).mean(0).numpy()
        eggs = []
        for n, ((score, box), p) in enumerate(zip(kept, probs), start=1):
            k = int(p.argmax())
            eggs.append({"n": n, "species": CLASSES[k], "confidence": float(p[k]),
                         "review": bool(p[k] < REVIEW_BELOW), "detector_score": float(score),
                         "box": [round(v) for v in box],
                         "probs": {c: float(v) for c, v in zip(CLASSES, p)}})
        return {"width": width, "height": height, "eggs": eggs, "review_below": REVIEW_BELOW,
                "image": draw(img, eggs)}


def make_handler(pipeline):
    page = (STATIC / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload):
            self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/analyze":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_UPLOAD:
                return self._json(413, {"error": f"la foto debe pesar entre 1 byte y {MAX_UPLOAD // 2**20} MB"})
            try:
                self._json(200, pipeline.analyze(self.rfile.read(length)))
            except Exception as exc:  # bad or unsupported image: tell the page, keep serving
                self._json(400, {"error": f"no se pudo analizar la foto: {exc}"})

        def log_message(self, fmt, *args):
            print(f"[app] {self.address_string()} {fmt % args}")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this computer")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print("loading YOLO + EfficientNet-B0 (3 seeds)...")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Pipeline()))
    print(f"ready: open http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
