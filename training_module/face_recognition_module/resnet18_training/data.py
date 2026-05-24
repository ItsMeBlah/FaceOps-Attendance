from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SPLIT_DIRS = {
    "train": "train_data",
    "val": "val_data",
    "test": "test_data",
}


def _resolve_split(root: Path, split: str) -> Path:
    split_dir = root / SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected {split} split at {split_dir}")
    return split_dir


def make_transforms(
    image_size: int,
    train: bool,
    grayscale: bool = True,
    resize_size: Optional[int] = None,
) -> transforms.Compose:
    steps = []
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=1))
    if resize_size is not None:
        steps.append(transforms.Resize((resize_size, resize_size)))
    if train:
        steps.extend(
            [
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        steps.append(transforms.CenterCrop((image_size, image_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if grayscale else transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


def build_dataset(
    data_root: str | Path,
    split: str,
    image_size: int,
    grayscale: bool = True,
    resize_size: Optional[int] = None,
) -> datasets.ImageFolder:
    root = Path(data_root).expanduser().resolve()
    if split not in SPLIT_DIRS:
        raise ValueError(f"split must be one of {sorted(SPLIT_DIRS)}")
    return datasets.ImageFolder(
        root=_resolve_split(root, split),
        transform=make_transforms(
            image_size=image_size,
            train=(split == "train"),
            grayscale=grayscale,
            resize_size=resize_size,
        ),
    )


def build_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 64,
    resize_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
    include_test: bool = True,
    grayscale: bool = True,
    pin_memory: bool = True,
) -> Tuple[Dict[str, DataLoader], Mapping[str, object]]:
    train_dataset = build_dataset(
        data_root,
        "train",
        image_size=image_size, 
        grayscale=grayscale,
        resize_size=resize_size,
    )
    val_dataset = build_dataset(
        data_root,
        "val",
        image_size=image_size,
        grayscale=grayscale,
        resize_size=resize_size,
    )
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
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
        test_dataset = build_dataset(
            data_root,
            "test",
            image_size=image_size,
            grayscale=grayscale,
            resize_size=resize_size,
        )
        if test_dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError("test_data classes do not match train_data classes")
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    metadata = {
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": len(train_dataset.classes),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(loaders["test"].dataset) if "test" in loaders else 0,
    }
    return loaders, metadata
