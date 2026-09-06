from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .geo import latlng_to_unit_np

REQUIRED_LABEL_COLUMNS = ["filename", "country", "iso", "lat", "lng"]
EXPECTED_COUNTRIES = {
    "Belarus",
    "Finland",
    "France",
    "Germany",
    "Iceland",
    "Italy",
    "Norway",
    "Poland",
    "Spain",
    "Sweden",
    "Turkey",
    "United_Kingdom",
}


def read_labels(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_LABEL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Labels are missing columns: {sorted(missing)}")
    if frame["filename"].duplicated().any():
        duplicates = frame.loc[frame["filename"].duplicated(), "filename"].head().tolist()
        raise ValueError(f"Duplicate filenames in labels: {duplicates}")
    if frame[REQUIRED_LABEL_COLUMNS].isna().any().any():
        raise ValueError("Labels contain missing values")
    if not frame["lat"].between(-90.0, 90.0).all():
        raise ValueError("Latitude outside [-90, 90]")
    if not frame["lng"].between(-180.0, 180.0).all():
        raise ValueError("Longitude outside [-180, 180]")
    return frame.reset_index(drop=True)


def add_country_ids(
    frame: pd.DataFrame, mapping: dict[str, int] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    if mapping is None:
        mapping = {country: idx for idx, country in enumerate(sorted(frame["country"].unique()))}
    unknown = sorted(set(frame["country"]) - set(mapping))
    if unknown:
        raise ValueError(f"Countries absent from checkpoint mapping: {unknown}")
    result = frame.copy()
    result["country_id"] = result["country"].map(mapping).astype(np.int64)
    return result, mapping


def audit_dataset(
    data_root: str | Path,
    expected_holdout_count: int | None = 2400,
    require_all_countries: bool = True,
) -> dict[str, object]:
    root = Path(data_root)
    labels = read_labels(root / "train_labels.csv")
    train_files = sorted((root / "train").glob("*.jpg"))
    holdout_files = sorted((root / "holdout_public").glob("*.jpg"))
    train_names = {path.name for path in train_files}
    labelled_names = set(labels["filename"])
    missing = sorted(labelled_names - train_names)
    orphan = sorted(train_names - labelled_names)
    if missing:
        raise FileNotFoundError(f"{len(missing)} labelled images are missing; examples: {missing[:5]}")
    if orphan:
        raise ValueError(f"{len(orphan)} train images have no label; examples: {orphan[:5]}")
    if expected_holdout_count is not None and len(holdout_files) != expected_holdout_count:
        raise ValueError(
            f"Expected {expected_holdout_count} holdout images, found {len(holdout_files)}"
        )
    countries = set(labels["country"])
    if require_all_countries and countries != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Country set mismatch. Missing={sorted(EXPECTED_COUNTRIES - countries)}, "
            f"extra={sorted(countries - EXPECTED_COUNTRIES)}"
        )
    return {
        "label_rows": len(labels),
        "train_images": len(train_files),
        "holdout_images": len(holdout_files),
        "countries": sorted(countries),
        "country_counts": labels["country"].value_counts().sort_index().to_dict(),
    }


def make_folds(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
    kind: str = "random",
    spatial_target_group_size: int = 40,
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    country_ids = frame["country_id"].to_numpy()
    folds = np.full(len(frame), -1, dtype=np.int64)
    if kind == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(frame)), country_ids)
    elif kind == "spatial":
        groups = _make_spatial_groups(frame, n_splits, spatial_target_group_size, seed)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(frame)), country_ids, groups)
    else:
        raise ValueError("Split kind must be 'random' or 'spatial'")
    for fold, (_, validation_indices) in enumerate(iterator):
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise RuntimeError("Fold construction failed")
    return folds


def _make_spatial_groups(
    frame: pd.DataFrame, n_splits: int, target_size: int, seed: int
) -> np.ndarray:
    groups = np.full(len(frame), -1, dtype=np.int64)
    country_ids = frame["country_id"].to_numpy()
    offset = 0
    for country_id in sorted(np.unique(country_ids)):
        positions = np.flatnonzero(country_ids == country_id)
        xyz = latlng_to_unit_np(frame.iloc[positions][["lat", "lng"]].to_numpy())
        n_groups = min(len(positions), max(n_splits, round(len(positions) / target_size)))
        local = KMeans(n_clusters=n_groups, n_init=20, random_state=seed).fit_predict(xyz)
        groups[positions] = local + offset
        offset += n_groups
    return groups


class RandomHalfTurn:
    """Occasionally correct/inject the 180-degree capture artifact present in the data."""

    def __init__(self, probability: float = 0.15):
        self.probability = float(probability)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return image.rotate(180, expand=False)


def build_transforms(
    image_size: int,
    train: bool,
    rotation_probability: float = 0.15,
    normalization: str = "half",
) -> transforms.Compose:
    if normalization == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif normalization == "half":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        raise ValueError("normalization must be 'half' or 'imagenet'")

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            RandomHalfTurn(rotation_probability),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.90, 1.10),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(0.25, 0.25, 0.20, 0.04)], p=0.80
            ),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.12
            ),
            transforms.RandomAutocontrast(p=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.10, scale=(0.01, 0.07), ratio=(0.5, 2.0)),
        ]
    )


class GeoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        transform: transforms.Compose,
        with_targets: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.with_targets = with_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = self.image_dir / row["filename"]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result: dict[str, object] = {
            "image": self.transform(image),
            "filename": str(row["filename"]),
        }
        if self.with_targets:
            latlng = torch.tensor([row["lat"], row["lng"]], dtype=torch.float32)
            radians = torch.deg2rad(latlng)
            result.update(
                {
                    "latlng": latlng,
                    "xyz": torch.stack(
                        (
                            torch.cos(radians[0]) * torch.cos(radians[1]),
                            torch.cos(radians[0]) * torch.sin(radians[1]),
                            torch.sin(radians[0]),
                        )
                    ),
                    "country_id": torch.tensor(int(row["country_id"]), dtype=torch.long),
                    "fine_cell": torch.tensor(int(row["fine_cell"]), dtype=torch.long),
                    "coarse_cell": torch.tensor(int(row["coarse_cell"]), dtype=torch.long),
                }
            )
        return result


class CellBatchSampler(Sampler[list[int]]):
    """Sample repeated geocells so supervised contrastive loss has positives."""

    def __init__(
        self,
        cell_ids: Sequence[int],
        batch_size: int,
        samples_per_cell: int = 2,
        seed: int = 2026,
    ) -> None:
        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.samples_per_cell = int(samples_per_cell)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < self.samples_per_cell:
            raise ValueError("batch_size must be >= samples_per_cell")
        self.unique_cells = np.unique(self.cell_ids)
        self.members = {cell: np.flatnonzero(self.cell_ids == cell) for cell in self.unique_cells}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.cell_ids) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        cells_per_batch = max(1, self.batch_size // self.samples_per_cell)
        for _ in range(len(self)):
            replace_cells = len(self.unique_cells) < cells_per_batch
            chosen_cells = generator.choice(self.unique_cells, cells_per_batch, replace=replace_cells)
            batch: list[int] = []
            for cell in chosen_cells:
                candidates = self.members[int(cell)]
                sampled = generator.choice(
                    candidates,
                    self.samples_per_cell,
                    replace=len(candidates) < self.samples_per_cell,
                )
                batch.extend(int(item) for item in sampled)
            while len(batch) < self.batch_size:
                batch.append(int(generator.integers(0, len(self.cell_ids))))
            generator.shuffle(batch)
            yield batch[: self.batch_size]


def suggested_num_workers(configured: int | str) -> int:
    if configured != "auto":
        return max(0, int(configured))
    return min(8, max(0, (os.cpu_count() or 2) - 1))


def make_loader(
    frame: pd.DataFrame,
    image_dir: str | Path,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
    rotation_probability: float,
    normalization: str,
    seed: int,
    samples_per_cell: int = 2,
) -> DataLoader:
    dataset = GeoDataset(
        frame,
        image_dir,
        build_transforms(image_size, train, rotation_probability, normalization),
        with_targets=True,
    )
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if train:
        sampler = CellBatchSampler(
            frame["fine_cell"].to_numpy(),
            batch_size=batch_size,
            samples_per_cell=samples_per_cell,
            seed=seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)
