from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .data import SPLIT_DIRS, build_dataset


class TripletFaceDataset(Dataset):
    """Wrap an ImageFolder dataset and sample identity-labelled face triplets."""

    def __init__(self, dataset: ImageFolder, train: bool, seed: int = 42):
        self.dataset = dataset
        self.train = train
        self.seed = seed

        samples_by_class: dict[int, list[int]] = defaultdict(list)
        for sample_idx, (_, label) in enumerate(dataset.samples):
            samples_by_class[int(label)].append(sample_idx)

        self.samples_by_class = {
            label: sample_indices
            for label, sample_indices in samples_by_class.items()
            if len(sample_indices) >= 2
        }
        self.eligible_labels = sorted(self.samples_by_class)
        if len(self.eligible_labels) < 2:
            raise ValueError("Triplet sampling requires at least two identities with two or more images each")

        self.anchor_indices = [
            sample_idx
            for label in self.eligible_labels
            for sample_idx in self.samples_by_class[label]
        ]
        self.num_ignored_identities = len(samples_by_class) - len(self.eligible_labels)

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, index: int):
        rng = random if self.train else random.Random(self.seed + index)
        anchor_idx = self.anchor_indices[index]
        anchor_label = int(self.dataset.samples[anchor_idx][1])

        positive_candidates = [
            sample_idx
            for sample_idx in self.samples_by_class[anchor_label]
            if sample_idx != anchor_idx
        ]
        positive_idx = rng.choice(positive_candidates)

        negative_label = rng.choice(self.eligible_labels)
        while negative_label == anchor_label:
            negative_label = rng.choice(self.eligible_labels)
        negative_idx = rng.choice(self.samples_by_class[negative_label])

        anchor = self.dataset[anchor_idx][0]
        positive = self.dataset[positive_idx][0]
        negative = self.dataset[negative_idx][0]
        return anchor, positive, negative, anchor_label, negative_label


def build_triplet_dataset(
    data_root: str | Path,
    split: str,
    image_size: int,
    grayscale: bool = True,
    resize_size: Optional[int] = None,
    seed: int = 42,
) -> TripletFaceDataset:
    dataset = build_dataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        grayscale=grayscale,
        resize_size=resize_size,
    )
    return TripletFaceDataset(dataset, train=(split == "train"), seed=seed)


def build_triplet_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 128,
    resize_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
    include_test: bool = True,
    grayscale: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Mapping[str, object]]:
    train_dataset = build_triplet_dataset(
        data_root,
        "train",
        image_size=image_size,
        grayscale=grayscale,
        resize_size=resize_size,
        seed=seed,
    )
    val_dataset = build_triplet_dataset(
        data_root,
        "val",
        image_size=image_size,
        grayscale=grayscale,
        resize_size=resize_size,
        seed=seed,
    )
    if val_dataset.dataset.class_to_idx != train_dataset.dataset.class_to_idx:
        raise ValueError("val_data classes do not match train_data classes")

    eval_batch_size = val_batch_size or batch_size
    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    test_dir = Path(data_root).expanduser().resolve() / SPLIT_DIRS["test"]
    if include_test and test_dir.is_dir():
        test_dataset = build_triplet_dataset(
            data_root,
            "test",
            image_size=image_size,
            grayscale=grayscale,
            resize_size=resize_size,
            seed=seed,
        )
        if test_dataset.dataset.class_to_idx != train_dataset.dataset.class_to_idx:
            raise ValueError("test_data classes do not match train_data classes")
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    metadata = {
        "classes": train_dataset.dataset.classes,
        "class_to_idx": train_dataset.dataset.class_to_idx,
        "num_classes": len(train_dataset.dataset.classes),
        "num_train_triplets": len(train_dataset),
        "num_val_triplets": len(val_dataset),
        "num_test_triplets": len(loaders["test"].dataset) if "test" in loaders else 0,
        "num_train_eligible_identities": len(train_dataset.eligible_labels),
        "num_val_eligible_identities": len(val_dataset.eligible_labels),
        "num_test_eligible_identities": len(loaders["test"].dataset.eligible_labels) if "test" in loaders else 0,
        "num_train_ignored_identities": train_dataset.num_ignored_identities,
        "num_val_ignored_identities": val_dataset.num_ignored_identities,
        "num_test_ignored_identities": loaders["test"].dataset.num_ignored_identities if "test" in loaders else 0,
    }
    return loaders, metadata
