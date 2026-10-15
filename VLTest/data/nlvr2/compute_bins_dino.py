import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T


# =========================
# 1. Load DINO
# =========================


def load_dino(device):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval()
    model.to(device)
    return model


# =========================
# 2. Image preprocessing
# =========================


def build_transform():
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# =========================
# 3. Collect train images
# =========================


def collect_train_images(train_root, max_images=None):
    exts = ("*.jpg", "*.png", "*.jpeg")
    image_paths = []
    for ext in exts:
        image_paths.extend(
            glob.glob(os.path.join(train_root, "**", ext), recursive=True)
        )

    image_paths = sorted(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    return image_paths


# =========================
# 4. Extract train features
# =========================


@torch.no_grad()
def extract_train_features(image_paths, model, transform, device, batch_size=32):
    features = []

    batch = []
    for img_path in tqdm(image_paths, desc="Extracting train features"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        batch.append(transform(img))

        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            feats = model(x).cpu().numpy()
            features.append(feats)
            batch.clear()

    if len(batch) > 0:
        x = torch.stack(batch).to(device)
        feats = model(x).cpu().numpy()
        features.append(feats)

    features = np.concatenate(features, axis=0)
    return features  # [N_train, d]


# =========================
# 5. Compute bins
# =========================


def compute_bins(train_features, B):
    """
    train_features: [N, d]
    """
    N, d = train_features.shape
    print(f"Computing bins from {N} samples, dim={d}, B={B}")

    bins = []
    qs = np.linspace(0, 1, B + 1)

    for k in range(d):
        bk = np.quantile(train_features[:, k], qs)
        bins.append(bk)

    return bins


# =========================
# 6. Main
# =========================


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading DINO...")
    model = load_dino(device)
    transform = build_transform()

    print("Collecting train images...")
    train_images = collect_train_images(args.train_root, max_images=args.max_images)
    print(f"Total train images used: {len(train_images)}")

    print("Extracting train features...")
    train_features = extract_train_features(
        train_images, model, transform, device, batch_size=args.batch_size
    )

    print("Computing bins...")
    bins = compute_bins(train_features, args.num_bins)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"bins_dino_B{args.num_bins}.npz")

    np.savez(save_path, bins=bins)

    print(f"Saved bins to {save_path}")
    print("Done.")


# =========================
# 7. CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_root",
        type=str,
        default="./train/images_left",
        help="Root directory of training images (e.g., COCO train2017)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./features", help="Directory to save bins"
    )
    parser.add_argument(
        "--num_bins", type=int, default=10, help="Number of bins per feature dimension"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit on number of train images",
    )

    args = parser.parse_args()
    main(args)
