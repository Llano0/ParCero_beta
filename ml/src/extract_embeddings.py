"""Turn every egg crop into a DINOv2 embedding vector.

Reads data/crops/crops.csv (from crop_eggs.py) and writes one .npz with:
    cls         (N, D)  CLS token — the standard global image embedding
    patch_mean  (N, D)  mean of the patch tokens — alternative global embedding
    labels, splits, groups, crop_paths  (N,)  aligned metadata

Usage:
    python ml/src/extract_embeddings.py --model dinov2_vitb14
    # other sizes: dinov2_vits14 (384-d), dinov2_vitl14 (1024-d)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pad_to_square(img):
    """Pad with the crop's mean color instead of stretching: egg shape (round vs.
    elongated) is discriminative, so the aspect ratio must survive the resize."""
    w, h = img.size
    side = max(w, h)
    fill = tuple(int(c) for c in np.asarray(img).reshape(-1, 3).mean(axis=0))
    return ImageOps.pad(img, (side, side), color=fill, centering=(0.5, 0.5))


class CropDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-csv", type=Path, default=Path("data/crops/crops.csv"))
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--size", type=int, default=224, help="input side, must be a multiple of 14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("ml/outputs/embeddings"))
    args = ap.parse_args()

    df = pd.read_csv(args.crops_csv)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.model).eval().to(device)

    loader = DataLoader(CropDataset(df["crop_path"].tolist(), args.size), batch_size=args.batch, num_workers=0)
    cls_list, patch_list = [], []
    with torch.inference_mode():
        for b, x in enumerate(loader):
            out = model.forward_features(x.to(device))
            cls_list.append(out["x_norm_clstoken"].cpu().numpy())
            patch_list.append(out["x_norm_patchtokens"].mean(dim=1).cpu().numpy())
            print(f"\r{min((b + 1) * args.batch, len(df))}/{len(df)}", end="", flush=True)
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.model}_{args.size}.npz"
    np.savez_compressed(
        out_path,
        cls=np.concatenate(cls_list).astype(np.float32),
        patch_mean=np.concatenate(patch_list).astype(np.float32),
        labels=df["label"].to_numpy(),
        splits=df["split"].to_numpy(),
        groups=df["group"].to_numpy(),
        crop_paths=df["crop_path"].to_numpy(),
    )
    print(f"Saved {out_path}  cls={cls_list[0].shape[1]}-d  n={len(df)}  device={device}")


if __name__ == "__main__":
    main()
