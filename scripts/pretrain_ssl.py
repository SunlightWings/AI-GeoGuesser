#!/usr/bin/env python3
"""SimCLR pretraining on a supplied-data label subset; no holdout or external images."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from geoguesser.data import build_transforms, read_labels, suggested_num_workers
from geoguesser.model import SignedGeM, build_backbone
from geoguesser.utils import choose_device, seed_everything


class TwoViewDataset(Dataset):
    def __init__(self, filenames: list[str], image_dir: Path, image_size: int) -> None:
        self.filenames = filenames
        self.image_dir = image_dir
        self.transform = build_transforms(
            image_size, train=True, rotation_probability=0.50, normalization="half"
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_dir / self.filenames[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), self.transform(image)


class SimCLR(nn.Module):
    def __init__(self, backbone_name: str, projection_dim: int = 128) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=False)
        self.pool = SignedGeM()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.pool(self.backbone(image))), dim=1)


def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = len(first)
    features = torch.cat((first, second), dim=0)
    logits = features @ features.T / temperature
    logits.fill_diagonal_(-1e4)
    targets = (torch.arange(2 * batch_size, device=features.device) + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--labels",
        default=None,
        type=Path,
        help="Label subset to pretrain on; use fold_N/train_rows.csv for honest cross-validation",
    )
    parser.add_argument("--backbone", choices=["efficientnet_b0", "mobilenet_v3_small"], default="efficientnet_b0")
    parser.add_argument("--output", default="outputs/ssl/backbone.pt", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("SimCLR requires batch_size >= 2")
    seed_everything(args.seed)
    device = choose_device(args.device)
    labels_path = (args.labels or (args.data_root / "train_labels.csv")).resolve()
    frame = read_labels(labels_path)
    missing = [name for name in frame["filename"] if not (args.data_root / "train" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SSL images: {missing[:5]}")
    dataset = TwoViewDataset(frame["filename"].tolist(), args.data_root / "train", args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=suggested_num_workers(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = SimCLR(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for first, second in tqdm(loader, desc=f"ssl {epoch:03d}", leave=False):
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = nt_xent(model(first), model(second), args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)
        scheduler.step()
        mean_loss = total / max(1, len(loader))
        history.append({"epoch": epoch, "loss": mean_loss})
        print(history[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "backbone": args.backbone,
            "backbone_state": model.backbone.state_dict(),
            "source": "provided training images only",
            "labels": str(labels_path),
            "epochs": args.epochs,
            "history": history,
            "elapsed_seconds": time.perf_counter() - start,
        },
        args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
