from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NormalizedEmbeddingModel, resnet18_face
from .triplet_data import build_triplet_dataloaders


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
class TripletTrainConfig:
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_dir: str = "checkpoints_resnet18_triplet"
    image_size: int = 128
    resize_size: int = 144
    embedding_size: int = 512
    batch_size: int = 64
    val_batch_size: int = 128
    epochs: int = 30
    num_workers: int = 4
    margin: float = 0.3
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    momentum: float = 0.9
    scheduler: str = "cosine"
    lr_step: int = 10
    lr_gamma: float = 0.2
    min_lr: float = 1e-6
    use_se: bool = False
    dropout: float = 0.35
    freeze_backbone: int = 5
    amp: bool = False
    seed: int = 42
    progress: bool = True
    plot_logs: bool = True
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    save_every: int = 1
    resume: str = ""
    resume_training_state: bool = False
    pretrained_backbone: str = ""
    no_cuda: bool = False


class TripletLogger:
    def __init__(self, output_dir: Path, enabled_plots: bool = True):
        self.output_dir = output_dir
        self.enabled_plots = enabled_plots
        self.history: list[Dict[str, object]] = []
        self.metrics_path = output_dir / "metrics.csv"
        self.history_path = output_dir / "training_history.json"

    def log_epoch(self, epoch: int, lr: float, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        row: Dict[str, object] = {"epoch": epoch, "lr": lr}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        self.history.append(row)
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        with self.history_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        self.plot_history()

    def plot_history(self) -> None:
        if not self.enabled_plots or not self.history:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipped plot generation")
            return

        epochs = [int(row["epoch"]) for row in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self._plot(axes[0, 0], epochs, "loss", "Triplet Loss")
        self._plot(axes[0, 1], epochs, "triplet_accuracy", "Triplet Accuracy")
        self._plot(axes[1, 0], epochs, "positive_distance", "Positive Distance")
        self._plot(axes[1, 0], epochs, "negative_distance", "Negative Distance")
        axes[1, 0].legend()
        axes[1, 1].plot(epochs, [row["lr"] for row in self.history], label="lr")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=160)
        plt.close(fig)

    def _plot(self, ax, epochs: list[int], metric: str, title: str) -> None:
        for split in ("train", "val"):
            key = f"{split}_{metric}"
            ax.plot(epochs, [row[key] for row in self.history], label=key)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TripletTrainer:
    def __init__(self, config: TripletTrainConfig):
        self.config = config
        set_seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loaders, self.metadata = build_triplet_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            resize_size=config.resize_size if config.resize_size > 0 else None,
            include_test=True,
            grayscale=True,
            pin_memory=(self.device.type == "cuda"),
            seed=config.seed,
        )

        backbone = resnet18_face(
            input_size=config.image_size,
            embedding_size=config.embedding_size,
            input_channels=1,
            use_se=config.use_se,
            dropout=config.dropout,
        )
        self.model = NormalizedEmbeddingModel(backbone).to(self.device)
        self.frozen_backbone_modules = self._freeze_backbone(config.freeze_backbone)
        self.criterion = nn.TripletMarginLoss(margin=config.margin, p=2)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.logger = TripletLogger(self.output_dir, enabled_plots=config.plot_logs)
        self.start_epoch = 1
        self.best_val_loss = float("inf")

        if config.pretrained_backbone and not config.resume:
            self.load_backbone_weights(config.pretrained_backbone)
        if config.resume:
            self.load_checkpoint(config.resume, load_training_state=config.resume_training_state)
        if config.freeze_backbone > 0 and not config.pretrained_backbone and not config.resume:
            print("warning: backbone is frozen without pretrained/resume weights")

        self._write_metadata()

    def _freeze_backbone(self, level: int) -> list[nn.Module]:
        if level < 0 or level > 5:
            raise ValueError("freeze_backbone must be between 0 and 5")
        if level == 0:
            return []

        backbone = self.model.backbone
        stages = [
            ("stem+layer1", [backbone.conv1, backbone.bn1, backbone.prelu, backbone.layer1]),
            ("layer2", [backbone.layer2]),
            ("layer3", [backbone.layer3]),
            ("layer4", [backbone.layer4]),
            ("embedding", [backbone.bn4, backbone.dropout, backbone.fc5, backbone.bn5]),
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

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params: Iterable[nn.Parameter] = [param for param in self.model.parameters() if param.requires_grad]
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
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step,
                gamma=self.config.lr_gamma,
            )
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        raise ValueError("scheduler must be 'step' or 'cosine'")

    def _progress(self, loader, description: str):
        if not self.config.progress or tqdm is None:
            return loader
        return tqdm(loader, desc=description, leave=False, dynamic_ncols=True)

    def _write_metadata(self) -> None:
        payload = {"config": asdict(self.config), "metadata": self.metadata}
        with (self.output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _forward_triplet(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        batch_size = anchor.size(0)
        embeddings = self.model(torch.cat([anchor, positive, negative], dim=0))
        return embeddings[:batch_size], embeddings[batch_size:2 * batch_size], embeddings[2 * batch_size:]

    def _run_epoch(self, split: str, epoch: int, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
            self._keep_frozen_backbone_eval()
        else:
            self.model.eval()

        total_loss = 0.0
        total_positive_distance = 0.0
        total_negative_distance = 0.0
        total_correct = 0
        total_active = 0
        total_seen = 0

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            for anchor, positive, negative, _, _ in self._progress(self.loaders[split], f"epoch {epoch:03d} {split}"):
                anchor = anchor.to(self.device, non_blocking=True)
                positive = positive.to(self.device, non_blocking=True)
                negative = negative.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.config.amp and self.device.type == "cuda"):
                    anchor_embedding, positive_embedding, negative_embedding = self._forward_triplet(anchor, positive, negative)
                    loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                positive_distance = F.pairwise_distance(anchor_embedding.detach(), positive_embedding.detach())
                negative_distance = F.pairwise_distance(anchor_embedding.detach(), negative_embedding.detach())
                batch_size = anchor.size(0)
                total_loss += loss.item() * batch_size
                total_positive_distance += positive_distance.sum().item()
                total_negative_distance += negative_distance.sum().item()
                total_correct += (positive_distance + self.config.margin < negative_distance).sum().item()
                total_active += (positive_distance - negative_distance + self.config.margin > 0).sum().item()
                total_seen += batch_size

        return {
            "loss": total_loss / max(total_seen, 1),
            "triplet_accuracy": total_correct / max(total_seen, 1),
            "positive_distance": total_positive_distance / max(total_seen, 1),
            "negative_distance": total_negative_distance / max(total_seen, 1),
            "active_triplet_ratio": total_active / max(total_seen, 1),
        }

    def run(self) -> None:
        print(f"device: {self.device}")
        print(
            f"train triplets: {self.metadata['num_train_triplets']} | "
            f"val triplets: {self.metadata['num_val_triplets']} | "
            f"test triplets: {self.metadata['num_test_triplets']}"
        )
        print(
            f"eligible identities: train {self.metadata['num_train_eligible_identities']} | "
            f"val {self.metadata['num_val_eligible_identities']} | "
            f"test {self.metadata['num_test_eligible_identities']}"
        )

        epochs_without_improvement = 0
        last_epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, self.config.epochs + 1):
                last_epoch = epoch
                train_metrics = self._run_epoch("train", epoch, training=True)
                val_metrics = self._run_epoch("val", epoch, training=False)
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_epoch(epoch, lr, train_metrics, val_metrics)
                self.scheduler.step()

                print(
                    f"epoch {epoch:03d} done | "
                    f"train loss {train_metrics['loss']:.4f} acc {train_metrics['triplet_accuracy']:.4f} | "
                    f"val loss {val_metrics['loss']:.4f} acc {val_metrics['triplet_accuracy']:.4f} "
                    f"pos_dist {val_metrics['positive_distance']:.4f} neg_dist {val_metrics['negative_distance']:.4f} "
                    f"active {val_metrics['active_triplet_ratio']:.4f}"
                )

                val_loss = float(val_metrics["loss"])
                is_best = val_loss < self.best_val_loss
                meaningful_improvement = val_loss < self.best_val_loss - max(self.config.early_stopping_min_delta, 0.0)
                if is_best:
                    self.best_val_loss = val_loss
                if is_best or (self.config.save_every and epoch % self.config.save_every == 0):
                    self.save_checkpoint(epoch, is_best=is_best)

                if self.config.early_stopping_patience > 0:
                    epochs_without_improvement = 0 if meaningful_improvement else epochs_without_improvement + 1
                    if epochs_without_improvement >= self.config.early_stopping_patience:
                        print(f"early stopping at epoch {epoch:03d}: best val loss {self.best_val_loss:.4f}")
                        break

            if "test" in self.loaders:
                test_metrics = self._run_epoch("test", last_epoch, training=False)
                print(
                    f"test loss {test_metrics['loss']:.4f} acc {test_metrics['triplet_accuracy']:.4f} "
                    f"pos_dist {test_metrics['positive_distance']:.4f} neg_dist {test_metrics['negative_distance']:.4f} "
                    f"active {test_metrics['active_triplet_ratio']:.4f}"
                )
        finally:
            self.logger.close()

    def checkpoint_payload(self, epoch: int) -> Dict[str, object]:
        return {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "model_state": self.model.backbone.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": asdict(self.config),
        }

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = self.checkpoint_payload(epoch)
        torch.save(payload, self.output_dir / "last.pth")
        if is_best:
            torch.save(payload, self.output_dir / "best.pth")
        if self.config.save_every and epoch % self.config.save_every == 0:
            torch.save(payload, self.output_dir / f"epoch_{epoch:03d}.pth")

    def load_checkpoint(self, checkpoint_path: str, load_training_state: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.model.backbone.load_state_dict(self._strip_module_prefix(state_dict))
        if load_training_state:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
                self.start_epoch = int(checkpoint["epoch"]) + 1
                print(f"resumed full triplet training state from {checkpoint_path} at epoch {self.start_epoch}")
                return
            except (KeyError, ValueError) as exc:
                print(f"warning: could not restore optimizer/scheduler state: {exc}")
        self.best_val_loss = float("inf")
        self.start_epoch = 1
        print(f"loaded backbone weights from {checkpoint_path}; starting a fresh triplet fine-tune run")

    def load_backbone_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._strip_module_prefix(checkpoint.get("model_state", checkpoint))
        model_state = self.model.backbone.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        model_state.update(compatible)
        self.model.backbone.load_state_dict(model_state)
        print(f"loaded {len(compatible)} backbone tensors from {checkpoint_path}; skipped {len(state_dict) - len(compatible)}")

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(key.startswith("module.") for key in state_dict):
            return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        return state_dict


def parse_args() -> TripletTrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet18 face embeddings with TripletMarginLoss.")
    parser.add_argument("--data-root", default=TripletTrainConfig.data_root)
    parser.add_argument("--output-dir", default=TripletTrainConfig.output_dir)
    parser.add_argument("--image-size", type=int, default=TripletTrainConfig.image_size)
    parser.add_argument("--resize-size", type=int, default=TripletTrainConfig.resize_size)
    parser.add_argument("--embedding-size", type=int, default=TripletTrainConfig.embedding_size)
    parser.add_argument("--batch-size", type=int, default=TripletTrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TripletTrainConfig.val_batch_size)
    parser.add_argument("--epochs", type=int, default=TripletTrainConfig.epochs)
    parser.add_argument("--num-workers", type=int, default=TripletTrainConfig.num_workers)
    parser.add_argument("--margin", type=float, default=TripletTrainConfig.margin)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=TripletTrainConfig.optimizer)
    parser.add_argument("--lr", type=float, default=TripletTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TripletTrainConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=TripletTrainConfig.momentum)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default=TripletTrainConfig.scheduler)
    parser.add_argument("--lr-step", type=int, default=TripletTrainConfig.lr_step)
    parser.add_argument("--lr-gamma", type=float, default=TripletTrainConfig.lr_gamma)
    parser.add_argument("--min-lr", type=float, default=TripletTrainConfig.min_lr)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--dropout", type=float, default=TripletTrainConfig.dropout)
    parser.add_argument("--freeze-backbone", type=int, choices=range(0, 6), default=TripletTrainConfig.freeze_backbone)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=TripletTrainConfig.seed)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument("--no-plots", action="store_false", dest="plot_logs")
    parser.add_argument("--early-stopping-patience", type=int, default=TripletTrainConfig.early_stopping_patience)
    parser.add_argument("--early-stopping-min-delta", type=float, default=TripletTrainConfig.early_stopping_min_delta)
    parser.add_argument("--save-every", type=int, default=TripletTrainConfig.save_every)
    parser.add_argument("--resume", default=TripletTrainConfig.resume)
    parser.add_argument("--resume-training-state", action="store_true")
    parser.add_argument("--pretrained-backbone", default=TripletTrainConfig.pretrained_backbone)
    parser.add_argument("--no-cuda", action="store_true")
    parser.set_defaults(progress=TripletTrainConfig.progress)
    parser.set_defaults(plot_logs=TripletTrainConfig.plot_logs)
    return TripletTrainConfig(**vars(parser.parse_args()))


def main() -> None:
    TripletTrainer(parse_args()).run()


if __name__ == "__main__":
    main()
