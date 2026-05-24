from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import build_dataloaders
from .heads import ArcMarginProduct
from .losses import FocalLoss
from .model import resnet18_face


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "11-785-fall-20-homework-2-part-2"
    / "classification_data_aligned"
)


@dataclass
class TrainConfig:
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_dir: str = "checkpoints_resnet18"
    image_size: int = 128 # 112
    resize_size: int = 144
    embedding_size: int = 512
    batch_size: int = 64
    val_batch_size: int = 256
    epochs: int = 50
    num_workers: int = 4
    head: str = "arcface"
    loss: str = "focal"
    optimizer: str = "sgd"
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    scheduler: str = "step"
    lr_step: int = 10
    lr_gamma: float = 0.1
    min_lr: float = 1e-6
    arc_scale: float = 30.0
    arc_margin: float = 0.5
    easy_margin: bool = False
    focal_gamma: float = 2.0
    use_se: bool = False
    dropout: float = 0.35
    freeze_backbone: int = 4
    amp: bool = False
    seed: int = 42
    progress: bool = True
    plot_logs: bool = True
    confusion_image_max_classes: int = 200
    verification_eval: bool = True
    verification_negative_ratio: int = 5
    verification_max_positive_pairs: int = 20000
    verification_max_negative_pairs: int = 100000
    verification_save_pairs: bool = True
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    save_every: int = 1
    resume: str = ""
    resume_training_state: bool = False
    pretrained_backbone: str = "/home/minhcao/Swinburne/COS30082/CustomProject/-Facial-Recognition-with-Emotion-and-Liveness/training_module/face_recognition_module/weights/resnet18_110.pth"
    no_cuda: bool = False


class MetricTracker:
    def __init__(self, num_classes: int, build_confusion: bool = False):
        self.num_classes = num_classes
        self.build_confusion = build_confusion
        self.total_seen = 0
        self.total_correct = 0
        self.tp = torch.zeros(num_classes, dtype=torch.long)
        self.pred_count = torch.zeros(num_classes, dtype=torch.long)
        self.target_count = torch.zeros(num_classes, dtype=torch.long)
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long) if build_confusion else None

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()
        correct = preds.eq(targets)

        self.total_seen += targets.numel()
        self.total_correct += correct.sum().item()
        self.tp += torch.bincount(targets[correct], minlength=self.num_classes)
        self.pred_count += torch.bincount(preds, minlength=self.num_classes)
        self.target_count += torch.bincount(targets, minlength=self.num_classes)

        if self.confusion is not None:
            indices = targets * self.num_classes + preds
            self.confusion += torch.bincount(indices, minlength=self.num_classes * self.num_classes).view(
                self.num_classes,
                self.num_classes,
            )

    def compute(self) -> Dict[str, float]:
        tp = self.tp.float()
        pred_count = self.pred_count.float()
        target_count = self.target_count.float()
        precision_per_class = torch.where(pred_count > 0, tp / pred_count.clamp_min(1), torch.zeros_like(tp))
        recall_per_class = torch.where(target_count > 0, tp / target_count.clamp_min(1), torch.zeros_like(tp))
        f1_per_class = torch.where(
            (precision_per_class + recall_per_class) > 0,
            2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class).clamp_min(1e-12),
            torch.zeros_like(tp),
        )
        supported = target_count > 0
        supported_count = max(int(supported.sum().item()), 1)

        return {
            "accuracy": self.total_correct / max(self.total_seen, 1),
            "precision": precision_per_class[supported].sum().item() / supported_count,
            "recall": recall_per_class[supported].sum().item() / supported_count,
            "f1": f1_per_class[supported].sum().item() / supported_count,
        }


class TrainingLogger:
    def __init__(self, output_dir: Path, enabled_plots: bool = True):
        self.output_dir = output_dir
        self.metrics_path = output_dir / "metrics.csv"
        self.history_path = output_dir / "training_history.json"
        self.enabled_plots = enabled_plots
        self.history: list[Dict[str, object]] = []

    def log_epoch(self, epoch: int, lr: float, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, "lr": lr}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        self.history.append(row)
        self._append_csv(row)

    def log_test(self, metrics: Dict[str, float], confusion: Optional[torch.Tensor], max_image_classes: int) -> None:
        row = {"epoch": "test", "lr": ""}
        row.update({f"test_{key}": value for key, value in metrics.items()})
        self.history.append(row)
        self._append_csv(row)
        if self.enabled_plots and confusion is not None and confusion.size(0) <= max_image_classes:
            self.plot_confusion_matrix(confusion)

    def _append_csv(self, row: Dict[str, object]) -> None:
        write_header = not self.metrics_path.exists()
        existing_fields = []
        if self.metrics_path.exists():
            with self.metrics_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])

        fields = list(dict.fromkeys([*existing_fields, *row.keys()]))
        if existing_fields and fields != existing_fields:
            with self.metrics_path.open("r", newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
            with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
            write_header = False

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_history(self) -> None:
        with self.history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def plot_history(self) -> None:
        if not self.enabled_plots:
            return
        epoch_rows = [row for row in self.history if isinstance(row.get("epoch"), int)]
        if not epoch_rows:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipped plot generation")
            return

        epochs = [int(row["epoch"]) for row in epoch_rows]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        self._plot_metric(axes[0], epochs, epoch_rows, "loss", "Loss")
        self._plot_metric(axes[1], epochs, epoch_rows, "accuracy", "Accuracy")
        self._plot_metric(axes[2], epochs, epoch_rows, "f1", "F1")
        self._plot_lr(axes[3], epochs, epoch_rows)

        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in ["precision", "recall", "f1", "accuracy"]:
            values = [row.get(f"val_{metric}") for row in epoch_rows]
            if any(value is not None for value in values):
                ax.plot(epochs, values, label=f"val_{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "validation_metrics.png", dpi=160)
        plt.close(fig)

    def _plot_metric(self, ax, epochs: list[int], rows: list[Dict[str, object]], metric: str, title: str) -> None:
        for split in ["train", "val"]:
            key = f"{split}_{metric}"
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                ax.plot(epochs, values, label=key)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_lr(self, ax, epochs: list[int], rows: list[Dict[str, object]]) -> None:
        values = [row.get("lr") for row in rows]
        ax.plot(epochs, values, label="lr")
        ax.set_xlabel("Epoch")
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_confusion_matrix(self, confusion: torch.Tensor) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipped confusion matrix plot")
            return

        image = confusion.float()
        image = image / image.max().clamp_min(1.0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.numpy(), interpolation="nearest", cmap="Blues")
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        fig.tight_layout()
        fig.savefig(self.output_dir / "test_confusion_matrix.png", dpi=160)
        plt.close(fig)

    def save_roc_curve(
        self,
        split: str,
        fpr: torch.Tensor,
        tpr: torch.Tensor,
        thresholds: torch.Tensor,
        auc: float,
    ) -> None:
        csv_path = self.output_dir / f"{split}_roc_curve.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "fpr", "tpr"])
            for threshold, fpr_value, tpr_value in zip(thresholds.tolist(), fpr.tolist(), tpr.tolist()):
                writer.writerow([threshold, fpr_value, tpr_value])

        if not self.enabled_plots:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipped ROC curve plot")
            return

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr.numpy(), tpr.numpy(), label=f"{split} ROC (AUC={auc:.4f})")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.5", label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{split.capitalize()} Face Verification ROC")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{split}_roc_curve.png", dpi=160)
        plt.close(fig)

    def save_verification_pairs(
        self,
        split: str,
        pairs: Sequence[tuple[int, int, int]],
        scores: torch.Tensor,
        class_names: Sequence[str],
        labels: torch.Tensor,
        paths: Optional[Sequence[str]],
    ) -> None:
        csv_path = self.output_dir / f"{split}_verification_pairs.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "left_index",
                    "right_index",
                    "left_class",
                    "right_class",
                    "is_same",
                    "similarity",
                    "left_path",
                    "right_path",
                ]
            )
            for (left_idx, right_idx, is_same), score in zip(pairs, scores.tolist()):
                left_label = int(labels[left_idx].item())
                right_label = int(labels[right_idx].item())
                writer.writerow(
                    [
                        left_idx,
                        right_idx,
                        class_names[left_label],
                        class_names[right_label],
                        is_same,
                        score,
                        paths[left_idx] if paths is not None else "",
                        paths[right_idx] if paths is not None else "",
                    ]
                )

    def close(self) -> None:
        self.save_history()
        self.plot_history()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loaders, self.metadata = build_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            resize_size=config.resize_size if config.resize_size > 0 else None,
            include_test=True,
            grayscale=True,
            pin_memory=(self.device.type == "cuda"),
        )
        self.num_classes = int(self.metadata["num_classes"])

        self.model = resnet18_face(
            input_size=config.image_size,
            embedding_size=config.embedding_size,
            input_channels=1,
            use_se=config.use_se,
            dropout=config.dropout,
        ).to(self.device)
        self.frozen_backbone_modules = self._freeze_backbone(config.freeze_backbone)
        self.head = self._build_head().to(self.device)
        self.criterion = self._build_loss().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.logger = TrainingLogger(self.output_dir, enabled_plots=config.plot_logs)
        self.start_epoch = 1
        self.best_val_acc = 0.0

        if config.pretrained_backbone and not config.resume:
            self.load_backbone_weights(config.pretrained_backbone)
        if config.resume:
            self.load_checkpoint(config.resume, load_training_state=config.resume_training_state)
        if config.freeze_backbone > 0 and not config.pretrained_backbone and not config.resume:
            print("warning: backbone is frozen without pretrained/resume weights; only the head will learn from random features")

        self._write_metadata()

    def _build_head(self) -> nn.Module:
        if self.config.head == "arcface":
            return ArcMarginProduct(
                in_features=self.config.embedding_size,
                out_features=self.num_classes,
                scale=self.config.arc_scale,
                margin=self.config.arc_margin,
                easy_margin=self.config.easy_margin,
            )
        if self.config.head == "linear":
            return nn.Linear(self.config.embedding_size, self.num_classes)
        raise ValueError("head must be 'arcface' or 'linear'")

    def _build_loss(self) -> nn.Module:
        if self.config.loss == "focal":
            return FocalLoss(gamma=self.config.focal_gamma)
        if self.config.loss == "ce":
            return nn.CrossEntropyLoss()
        raise ValueError("loss must be 'focal' or 'ce'")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params: Iterable[torch.nn.Parameter] = [
            param
            for param in list(self.model.parameters()) + list(self.head.parameters())
            if param.requires_grad
        ]
        if not params:
            raise ValueError("No trainable parameters remain after applying freeze_backbone")
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        raise ValueError("optimizer must be 'sgd' or 'adamw'")

    def _build_scheduler(self):
        if self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.lr_step, gamma=self.config.lr_gamma)
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        raise ValueError("scheduler must be 'step' or 'cosine'")

    def _freeze_backbone(self, level: int) -> list[nn.Module]:
        if level < 0 or level > 5:
            raise ValueError("freeze_backbone must be between 0 and 5")
        if level == 0:
            return []

        stages = [
            ("stem+layer1", [self.model.conv1, self.model.bn1, self.model.prelu, self.model.layer1]),
            ("layer2", [self.model.layer2]),
            ("layer3", [self.model.layer3]),
            ("layer4", [self.model.layer4]),
            ("embedding", [self.model.bn4, self.model.dropout, self.model.fc5, self.model.bn5]),
        ]
        frozen_modules: list[nn.Module] = []
        frozen_names = []
        for name, modules in stages[:level]:
            frozen_names.append(name)
            for module in modules:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                frozen_modules.append(module)

        print(f"freeze_backbone={level}: froze {', '.join(frozen_names)}")
        return frozen_modules

    def _keep_frozen_backbone_eval(self) -> None:
        for module in self.frozen_backbone_modules:
            module.eval()

    def _write_metadata(self) -> None:
        payload = {
            "config": asdict(self.config),
            "metadata": {
                "classes": self.metadata["classes"],
                "class_to_idx": self.metadata["class_to_idx"],
                "num_classes": self.metadata["num_classes"],
                "num_train": self.metadata["num_train"],
                "num_val": self.metadata["num_val"],
                "num_test": self.metadata["num_test"],
            },
        }
        with (self.output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _forward_logits(self, images: torch.Tensor, labels: torch.Tensor | None, training: bool) -> torch.Tensor:
        features = self.model(images)
        if isinstance(self.head, ArcMarginProduct):
            return self.head(features, labels if training else None)
        return self.head(features)

    def _progress(self, loader, description: str):
        if not self.config.progress or tqdm is None:
            return loader
        return tqdm(loader, desc=description, leave=False, dynamic_ncols=True)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.head.train()
        self._keep_frozen_backbone_eval()
        total_loss = 0.0
        total_seen = 0
        metrics = MetricTracker(self.num_classes)

        for images, labels in self._progress(self.loaders["train"], f"epoch {epoch:03d} train"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.config.amp and self.device.type == "cuda"):
                features = self.model(images)
                if isinstance(self.head, ArcMarginProduct):
                    logits = self.head(features, labels)
                    acc_logits = self.head(features.detach(), None)
                else:
                    logits = self.head(features)
                    acc_logits = logits.detach()
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            preds = acc_logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_seen += batch_size
            metrics.update(preds, labels)

        results = {"loss": total_loss / max(total_seen, 1)}
        results.update(metrics.compute())
        return results

    @torch.no_grad()
    def evaluate(self, split: str, build_confusion: bool = False) -> Dict[str, object]:
        self.model.eval()
        self.head.eval()
        total_loss = 0.0
        total_seen = 0
        metrics = MetricTracker(self.num_classes, build_confusion=build_confusion)

        for images, labels in self._progress(self.loaders[split], f"{split} eval"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()
            logits = self._forward_logits(images, labels, training=False)
            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_seen += batch_size
            metrics.update(preds, labels)

        results: Dict[str, object] = {"loss": total_loss / max(total_seen, 1)}
        results.update(metrics.compute())
        if build_confusion:
            results["confusion_matrix"] = metrics.confusion
        return results

    @torch.no_grad()
    def extract_embeddings(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        embeddings = []
        labels_list = []

        for images, labels in self._progress(self.loaders[split], f"{split} embeddings"):
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            embeddings.append(F.normalize(features, dim=1).cpu())
            labels_list.append(labels.long().cpu())

        return torch.cat(embeddings, dim=0), torch.cat(labels_list, dim=0)

    def evaluate_verification(self, split: str) -> Dict[str, object]:
        if not self.config.verification_eval:
            return {}

        embeddings, labels = self.extract_embeddings(split)
        pairs = self._build_verification_pairs(labels, split)
        if not pairs:
            print(f"{split} verification skipped: not enough positive and negative pairs")
            return {
                "verification_auc": float("nan"),
                "verification_best_threshold": float("nan"),
                "verification_best_accuracy": float("nan"),
                "verification_eer": float("nan"),
                "verification_positive_pairs": 0,
                "verification_negative_pairs": 0,
                "verification_pairs": 0,
            }

        scores = self._score_verification_pairs(embeddings, pairs)
        targets = torch.tensor([is_same for _, _, is_same in pairs], dtype=torch.long)
        roc = self._compute_roc_auc(scores, targets)
        if roc is None:
            print(f"{split} verification skipped: ROC needs both same-person and different-person pairs")
            return {
                "verification_auc": float("nan"),
                "verification_best_threshold": float("nan"),
                "verification_best_accuracy": float("nan"),
                "verification_eer": float("nan"),
                "verification_positive_pairs": int(targets.sum().item()),
                "verification_negative_pairs": int((targets == 0).sum().item()),
                "verification_pairs": int(targets.numel()),
            }

        fpr, tpr, thresholds, auc, best_threshold, best_accuracy, eer = roc
        self.logger.save_roc_curve(split, fpr, tpr, thresholds, auc)

        if self.config.verification_save_pairs:
            self.logger.save_verification_pairs(
                split=split,
                pairs=pairs,
                scores=scores,
                class_names=self.metadata["classes"],
                labels=labels,
                paths=self._split_paths(split, labels.numel()),
            )

        return {
            "verification_auc": auc,
            "verification_best_threshold": best_threshold,
            "verification_best_accuracy": best_accuracy,
            "verification_eer": eer,
            "verification_positive_pairs": int(targets.sum().item()),
            "verification_negative_pairs": int((targets == 0).sum().item()),
            "verification_pairs": int(targets.numel()),
        }

    def _split_paths(self, split: str, expected_count: int) -> Optional[list[str]]:
        dataset = self.loaders[split].dataset
        samples = getattr(dataset, "samples", None)
        if samples is None or len(samples) != expected_count:
            return None
        return [str(path) for path, _ in samples]

    def _build_verification_pairs(self, labels: torch.Tensor, split: str) -> list[tuple[int, int, int]]:
        labels_by_class: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels.tolist()):
            labels_by_class[int(label)].append(idx)

        positive_total = sum(len(indices) * (len(indices) - 1) // 2 for indices in labels_by_class.values())
        positive_cap = self.config.verification_max_positive_pairs
        positive_target = positive_total if positive_cap <= 0 else min(positive_total, positive_cap)
        if positive_target == 0:
            return []

        split_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(split))
        rng = random.Random(self.config.seed + split_offset)
        positive_pairs = self._sample_positive_pairs(labels_by_class, positive_total, positive_target, rng)

        num_items = labels.numel()
        all_pair_count = num_items * (num_items - 1) // 2
        negative_total = all_pair_count - positive_total
        negative_target = min(negative_total, positive_target * max(self.config.verification_negative_ratio, 0))
        if self.config.verification_max_negative_pairs > 0:
            negative_target = min(negative_target, self.config.verification_max_negative_pairs)
        if negative_target == 0:
            return []

        negative_pairs = self._sample_negative_pairs(labels, negative_target, rng)
        return [(left, right, 1) for left, right in positive_pairs] + [
            (left, right, 0) for left, right in negative_pairs
        ]

    def _sample_positive_pairs(
        self,
        labels_by_class: dict[int, list[int]],
        positive_total: int,
        positive_target: int,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        if positive_total <= positive_target:
            pairs = []
            for indices in labels_by_class.values():
                for left_pos, left_idx in enumerate(indices):
                    for right_idx in indices[left_pos + 1:]:
                        pairs.append((left_idx, right_idx))
            return pairs

        eligible_labels = [label for label, indices in labels_by_class.items() if len(indices) >= 2]
        weights = [len(labels_by_class[label]) * (len(labels_by_class[label]) - 1) // 2 for label in eligible_labels]
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while len(pairs) < positive_target:
            label = rng.choices(eligible_labels, weights=weights, k=1)[0]
            left_idx, right_idx = sorted(rng.sample(labels_by_class[label], 2))
            key = (left_idx, right_idx)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        return pairs

    def _sample_negative_pairs(
        self,
        labels: torch.Tensor,
        negative_target: int,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        label_values = labels.tolist()
        num_items = len(label_values)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        max_attempts = max(negative_target * 50, 1000)
        attempts = 0

        while len(pairs) < negative_target and attempts < max_attempts:
            left_idx, right_idx = sorted(rng.sample(range(num_items), 2))
            attempts += 1
            if label_values[left_idx] == label_values[right_idx]:
                continue
            key = (left_idx, right_idx)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)

        if len(pairs) == negative_target:
            return pairs

        for left_idx in range(num_items):
            for right_idx in range(left_idx + 1, num_items):
                if label_values[left_idx] == label_values[right_idx]:
                    continue
                key = (left_idx, right_idx)
                if key in seen:
                    continue
                pairs.append(key)
                if len(pairs) == negative_target:
                    return pairs
        return pairs

    def _score_verification_pairs(self, embeddings: torch.Tensor, pairs: Sequence[tuple[int, int, int]]) -> torch.Tensor:
        left_indices = torch.tensor([left for left, _, _ in pairs], dtype=torch.long)
        right_indices = torch.tensor([right for _, right, _ in pairs], dtype=torch.long)
        return (embeddings[left_indices] * embeddings[right_indices]).sum(dim=1)

    def _compute_roc_auc(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]]:
        positives = int(targets.sum().item())
        negatives = int((targets == 0).sum().item())
        if positives == 0 or negatives == 0:
            return None

        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order].float()
        sorted_targets = targets[order].float()
        true_positives = torch.cumsum(sorted_targets, dim=0)
        false_positives = torch.cumsum(1.0 - sorted_targets, dim=0)

        distinct = torch.ones_like(sorted_targets, dtype=torch.bool)
        distinct[:-1] = sorted_scores[:-1] != sorted_scores[1:]

        tpr = true_positives[distinct] / positives
        fpr = false_positives[distinct] / negatives
        thresholds = sorted_scores[distinct]

        zero = torch.tensor([0.0])
        tpr = torch.cat([zero, tpr.cpu()])
        fpr = torch.cat([zero, fpr.cpu()])
        thresholds = torch.cat([torch.tensor([float("inf")]), thresholds.cpu()])
        auc = float(torch.trapz(tpr, fpr).item())

        best_idx = int(torch.argmax(tpr[1:] - fpr[1:]).item()) + 1
        best_threshold = float(thresholds[best_idx].item())
        predictions = scores >= best_threshold
        best_accuracy = float(predictions.eq(targets.bool()).float().mean().item())

        eer_idx = int(torch.argmin(torch.abs(fpr - (1.0 - tpr))).item())
        eer = float(((fpr[eer_idx] + (1.0 - tpr[eer_idx])) / 2.0).item())
        return fpr, tpr, thresholds, auc, best_threshold, best_accuracy, eer

    def run(self) -> None:
        print(f"device: {self.device}")
        print(
            f"classes: {self.num_classes} | train: {self.metadata['num_train']} | "
            f"val: {self.metadata['num_val']} | test: {self.metadata['num_test']}"
        )

        try:
            epochs_without_improvement = 0
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.evaluate("val")
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_epoch(epoch, lr, train_metrics, val_metrics)
                self.scheduler.step()

                print(
                    f"epoch {epoch:03d} done | "
                    f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
                    f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
                    f"precision {val_metrics['precision']:.4f} recall {val_metrics['recall']:.4f} "
                    f"f1 {val_metrics['f1']:.4f}"
                )

                previous_best_val_acc = self.best_val_acc
                val_accuracy = float(val_metrics["accuracy"])
                min_delta = max(self.config.early_stopping_min_delta, 0.0)
                is_best = val_accuracy > self.best_val_acc
                is_meaningful_improvement = val_accuracy > previous_best_val_acc + min_delta
                if is_best:
                    self.best_val_acc = val_accuracy
                should_save = is_best or (self.config.save_every and epoch % self.config.save_every == 0)
                if should_save:
                    self.save_checkpoint(epoch=epoch, is_best=is_best)

                if self.config.early_stopping_patience > 0:
                    if is_meaningful_improvement:
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.early_stopping_patience:
                        print(
                            f"early stopping at epoch {epoch:03d}: "
                            f"val accuracy did not improve by at least {min_delta:.6f} for "
                            f"{self.config.early_stopping_patience} epoch(s). "
                            f"best val acc {self.best_val_acc:.4f}"
                        )
                        break

            if "test" in self.loaders:
                test_metrics = self.evaluate("test", build_confusion=True)
                confusion = test_metrics.pop("confusion_matrix")
                test_metrics.update(self.evaluate_verification("test"))
                self.save_confusion_matrix(confusion)
                self.logger.log_test(test_metrics, confusion, self.config.confusion_image_max_classes)
                test_auc_text = ""
                if "verification_auc" in test_metrics:
                    test_auc_text = f" auc {float(test_metrics['verification_auc']):.4f}"
                print(
                    f"test loss {test_metrics['loss']:.4f} acc {test_metrics['accuracy']:.4f} "
                    f"precision {test_metrics['precision']:.4f} recall {test_metrics['recall']:.4f} "
                    f"f1 {test_metrics['f1']:.4f}{test_auc_text}"
                )
        finally:
            self.logger.close()

    def checkpoint_payload(self, epoch: int) -> Dict[str, object]:
        return {
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "model_state": self.model.state_dict(),
            "head_state": self.head.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "classes": self.metadata["classes"],
            "class_to_idx": self.metadata["class_to_idx"],
        }

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = self.checkpoint_payload(epoch)
        last_path = self.output_dir / "last.pth"
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, self.output_dir / "best.pth")
        if self.config.save_every and epoch % self.config.save_every == 0:
            torch.save(payload, self.output_dir / f"epoch_{epoch:03d}.pth")

    def save_confusion_matrix(self, confusion: torch.Tensor) -> None:
        torch.save(confusion, self.output_dir / "test_confusion_matrix.pt")
        nonzero_path = self.output_dir / "test_confusion_matrix_nonzero.csv"
        classes = self.metadata["classes"]
        rows, cols = torch.nonzero(confusion, as_tuple=True)
        with nonzero_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_idx", "pred_idx", "true_class", "pred_class", "count"])
            for row, col in zip(rows.tolist(), cols.tolist()):
                writer.writerow([row, col, classes[row], classes[col], int(confusion[row, col].item())])

    def load_checkpoint(self, checkpoint_path: str, load_training_state: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.head.load_state_dict(checkpoint["head_state"])
        if load_training_state:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
                self.start_epoch = int(checkpoint["epoch"]) + 1
                print(f"resumed full training state from {checkpoint_path} at epoch {self.start_epoch}")
                return
            except ValueError as exc:
                print(f"warning: could not restore optimizer/scheduler state: {exc}")
                print("continuing with checkpoint weights and fresh optimizer/scheduler")

        self.best_val_acc = 0.0
        self.start_epoch = 1
        print(f"loaded checkpoint weights from {checkpoint_path}; starting a fresh fine-tune run")

    def load_backbone_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        model_state = self.model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        skipped = len(state_dict) - len(compatible)
        model_state.update(compatible)
        self.model.load_state_dict(model_state)
        print(f"loaded {len(compatible)} backbone tensors from {checkpoint_path}; skipped {skipped}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet18 ArcFace on classification_data")
    parser.add_argument("--data-root", default=TrainConfig.data_root)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument(
        "--resize-size",
        type=int,
        default=TrainConfig.resize_size,
        help="Resize before ArcFace-style crop. Use 0 to disable resizing.",
    )
    parser.add_argument("--embedding-size", type=int, default=TrainConfig.embedding_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--head", choices=["arcface", "linear"], default=TrainConfig.head)
    parser.add_argument("--loss", choices=["focal", "ce"], default=TrainConfig.loss)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=TrainConfig.optimizer)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=TrainConfig.momentum)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default=TrainConfig.scheduler)
    parser.add_argument("--lr-step", type=int, default=TrainConfig.lr_step)
    parser.add_argument("--lr-gamma", type=float, default=TrainConfig.lr_gamma)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--arc-scale", type=float, default=TrainConfig.arc_scale)
    parser.add_argument("--arc-margin", type=float, default=TrainConfig.arc_margin)
    parser.add_argument("--easy-margin", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=TrainConfig.focal_gamma)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument(
        "--freeze-backbone",
        type=int,
        choices=range(0, 6),
        default=TrainConfig.freeze_backbone,
        metavar="{0,1,2,3,4,5}",
        help=(
            "Freeze backbone stages from input forward: 0 none, 1 stem+layer1, "
            "2 +layer2, 3 +layer3, 4 +layer4, 5 +embedding"
        ),
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--no-plots", action="store_false", dest="plot_logs")
    parser.add_argument("--confusion-image-max-classes", type=int, default=TrainConfig.confusion_image_max_classes)
    parser.add_argument("--no-verification-eval", action="store_false", dest="verification_eval")
    parser.add_argument(
        "--verification-negative-ratio",
        type=int,
        default=TrainConfig.verification_negative_ratio,
        help="Number of sampled different-person pairs per same-person pair.",
    )
    parser.add_argument(
        "--verification-max-positive-pairs",
        type=int,
        default=TrainConfig.verification_max_positive_pairs,
        help="Maximum same-person pairs for ROC/AUC. Use 0 for all available pairs.",
    )
    parser.add_argument(
        "--verification-max-negative-pairs",
        type=int,
        default=TrainConfig.verification_max_negative_pairs,
        help="Maximum different-person pairs for ROC/AUC. Use 0 to only apply the ratio cap.",
    )
    parser.add_argument("--no-verification-save-pairs", action="store_false", dest="verification_save_pairs")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainConfig.early_stopping_patience,
        help="Stop when val accuracy has no meaningful improvement for this many epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=TrainConfig.early_stopping_min_delta,
        help="Minimum val accuracy improvement required to reset early stopping patience.",
    )
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--resume", default=TrainConfig.resume)
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help="Restore optimizer, scheduler, and epoch from checkpoint. Use only for exact same training setup.",
    )
    parser.add_argument("--pretrained-backbone", default=TrainConfig.pretrained_backbone)
    parser.add_argument("--no-cuda", action="store_true")
    parser.set_defaults(progress=TrainConfig.progress)
    parser.set_defaults(plot_logs=TrainConfig.plot_logs)
    parser.set_defaults(verification_eval=TrainConfig.verification_eval)
    parser.set_defaults(verification_save_pairs=TrainConfig.verification_save_pairs)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    Trainer(config).run()


if __name__ == "__main__":
    main()
