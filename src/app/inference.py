"""Inference pipeline shared by the app and the notebook.

YOLO finds the eggs in a photo, every box is cropped exactly like crop_eggs.py
(15 % margin), DINOv2 turns each crop into an embedding (same transforms as
extract_embeddings.py) and the classical classifier of the ML branch names the
species. Each egg also gets two trust signals used to say "consult an expert":

    prob        highest class probability of the classifier
    similarity  cosine similarity to the closest reference crop (train + valid);
                low = the crop does not look like anything seen in training
                (false alarm of the detector, new stain, new microscope...)

Everything runs offline: YOLO weights, the DINOv2 hub cache and the classifier
bundle are local files.
"""

import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "embeddings"))
sys.path.insert(0, str(ROOT / "src" / "data"))
from crop_eggs import expand  # noqa: E402
from extract_embeddings import IMAGENET_MEAN, IMAGENET_STD, pad_to_square  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "artifacts" / "yolo" / "h_single_aug" / "weights" / "best.pt"
DEFAULT_BUNDLE = ROOT / "artifacts" / "ml" / "egg_classifier.joblib"
DINOV2_MODEL = "dinov2_vitb14"
CROP_SIZE = 224
CROP_MARGIN = 0.15   # same as crop_eggs.py and detect_crops.py
DET_CONF = 0.25      # detection threshold decided in compare_detectors.ipynb
IMGSZ = 640          # size the detector was trained with
MIN_CROP = 16

STATUS_OK = "ok"
STATUS_DOUBTFUL = "doubtful"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dinov2(name=DINOV2_MODEL, device=None):
    """DINOv2 from the local torch.hub cache (no internet); GitHub only as a fallback."""
    warnings.filterwarnings("ignore", message="xFormers is not available")
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if local_repo.is_dir():
        model = torch.hub.load(str(local_repo), name, source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", name)
    return model.eval().to(device or pick_device())


CROP_TRANSFORM = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((CROP_SIZE, CROP_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def embed_crops(model, crops, batch=64):
    """CLS embedding of every PIL crop, (N, D) float32. Same preprocessing as extract_embeddings.py."""
    device = next(model.parameters()).device
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch):
            x = torch.stack([CROP_TRANSFORM(c.convert("RGB")) for c in crops[start:start + batch]])
            vectors.append(model.forward_features(x.to(device))["x_norm_clstoken"].cpu().numpy())
    return np.concatenate(vectors).astype(np.float32) if vectors else np.zeros((0, 768), np.float32)


def to_unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def closest_similarity(vectors, unit_reference):
    """Cosine similarity of every vector to its closest reference vector."""
    if len(vectors) == 0:
        return np.zeros(0)
    return (to_unit(vectors) @ unit_reference.T).max(axis=1)


def egg_status(prob, similarity, conf_threshold, similarity_threshold):
    """'ok' when the classifier is sure and the crop resembles training data, else 'doubtful'."""
    return STATUS_OK if prob >= conf_threshold and similarity >= similarity_threshold else STATUS_DOUBTFUL


class EggPipeline:
    """Photo -> list of eggs with box, species, probability, similarity and status."""

    def __init__(self, weights=DEFAULT_WEIGHTS, bundle_path=DEFAULT_BUNDLE, device=None):
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO  # imported here so the notebook can use the rest without it

        self.device = device or pick_device()
        self.detector = YOLO(str(weights))
        self.dinov2 = load_dinov2(device=self.device)
        self.bundle = joblib.load(bundle_path)
        self.classifier = self.bundle["model"]
        self.classes = list(self.classifier.classes_)
        self.reference = self.bundle["reference"].astype(np.float32)

    def analyze(self, image):
        image = image.convert("RGB")
        width, height = image.size
        result = self.detector.predict(image, imgsz=IMGSZ, conf=DET_CONF, device="cpu", verbose=False)[0]
        boxes, crops = [], []
        for box, score in sorted(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()),
                                 key=lambda d: -d[1]):
            x1, y1, x2, y2 = expand(*box, CROP_MARGIN, width, height)
            if min(x2 - x1, y2 - y1) < MIN_CROP:
                continue
            boxes.append((box, score))
            crops.append(image.crop((x1, y1, x2, y2)))
        if not crops:
            return []

        vectors = embed_crops(self.dinov2, crops)
        probs = self.classifier.predict_proba(vectors)
        similarity = closest_similarity(vectors, self.reference)
        eggs = []
        for (box, score), p, sim in zip(boxes, probs, similarity):
            best = int(p.argmax())
            reason = ("unfamiliar" if sim < self.bundle["similarity_threshold"]
                      else "between_species" if p[best] < self.bundle["conf_threshold"] else "")
            eggs.append({
                "box": [round(v, 1) for v in box],
                "det_score": round(float(score), 3),
                "species": self.classes[best],
                "prob": round(float(p[best]), 3),
                "probs": {c: round(float(v), 3) for c, v in zip(self.classes, p)},
                "similarity": round(float(sim), 3),
                "status": egg_status(p[best], sim, self.bundle["conf_threshold"],
                                     self.bundle["similarity_threshold"]),
                "reason": reason,  # why it is doubtful: "unfamiliar" crop or "between_species"
            })
        return eggs


def load_image(path):
    return Image.open(path).convert("RGB")
