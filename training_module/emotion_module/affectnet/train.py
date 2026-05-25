import torch
import torch.nn as nn
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model import build_model, get_mobilenet_norm_stats, unfreeze_backbone

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "AffectNet/Train"
TEST_DIR = "AffectNet/Test"
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS_FROZEN = 30
EPOCHS_TUNED = 35
LR_FROZEN = 1e-3
LR_TUNED = 1e-4
SAVE_PATH = "emotion.pth"
METRICS_PATH = "metrics/test_metrics.txt"
HISTORY_CSV_PATH = "metrics/training_history.csv"
METRICS_DIR = Path("metrics")

print(f"Using device: {DEVICE}")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

NORM_MEAN, NORM_STD = get_mobilenet_norm_stats()
print(f"MobileNetV2 normalize mean: {NORM_MEAN}")
print(f"MobileNetV2 normalize std : {NORM_STD}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# Data
full_train_base = datasets.ImageFolder(TRAIN_DIR)
full_train_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_train_eval = datasets.ImageFolder(TRAIN_DIR, transform=test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR,  transform=test_transform)
EMOTIONS = full_train_base.classes

num_samples = len(full_train_base)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size
train_idx_subset, val_idx_subset = random_split(
    range(num_samples), [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
train_indices = list(train_idx_subset.indices)
val_indices = list(val_idx_subset.indices)

train_dataset = Subset(full_train_aug, train_indices)
val_dataset = Subset(full_train_eval, val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Class weights / punish more on major classes
counts = [0] * len(EMOTIONS)
for _, label in full_train_base.samples:
    counts[label] += 1
total = sum(counts)
weights = torch.tensor(
    [total / (len(EMOTIONS) * c) for c in counts],
    dtype=torch.float
).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

# Training loop
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def run_training(model, optimizer, scheduler, epochs, phase_name):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*50}\n  {phase_name}\n{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            saved = "✅ saved"
        else:
            saved = ""

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} {saved}")

    print(f"Best val acc: {best_val_acc:.4f}")
    return history


# Plot
def plot_history(h1, h2):
    train_loss = h1["train_loss"] + h2["train_loss"]
    val_loss = h1["val_loss"] + h2["val_loss"]
    train_acc = h1["train_acc"] + h2["train_acc"]
    val_acc = h1["val_acc"] + h2["val_acc"]
    epochs = range(1, len(train_loss) + 1)
    split = len(h1["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Val")
    axes[0].axvline(x=split, color='gray', linestyle='--', label="Fine-tune starts")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Val")
    axes[1].axvline(x=split, color='gray', linestyle='--', label="Fine-tune starts")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("metrics/training_curves.png")
    print("Saved: metrics/training_curves.png")

    with open(HISTORY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "train_loss", "val_loss", "train_acc", "val_acc"])
        for i in range(len(h1["train_loss"])):
            writer.writerow([
                i + 1,
                "frozen",
                h1["train_loss"][i],
                h1["val_loss"][i],
                h1["train_acc"][i],
                h1["val_acc"][i],
            ])
        for i in range(len(h2["train_loss"])):
            writer.writerow([
                len(h1["train_loss"]) + i + 1,
                "fine_tune",
                h2["train_loss"][i],
                h2["val_loss"][i],
                h2["train_acc"][i],
                h2["val_acc"][i],
            ])
    print(f"Saved: {HISTORY_CSV_PATH}")


# Test evaluation
def evaluate_test(model):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report_text = classification_report(all_labels, all_preds, target_names=EMOTIONS, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n=== Classification Report ===")
    print(report_text)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("=== Test Metrics Summary ===\n")
        f.write(f"Accuracy           : {accuracy:.4f}\n")
        f.write(f"Macro Precision    : {prec_macro:.4f}\n")
        f.write(f"Macro Recall       : {rec_macro:.4f}\n")
        f.write(f"Macro F1           : {f1_macro:.4f}\n")
        f.write(f"Weighted Precision : {prec_weighted:.4f}\n")
        f.write(f"Weighted Recall    : {rec_weighted:.4f}\n")
        f.write(f"Weighted F1        : {f1_weighted:.4f}\n\n")
        f.write("=== Per-class Report ===\n")
        f.write(report_text)
    print(f"Saved: {METRICS_PATH}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png")
    print("Saved: metrics/confusion_matrix.png")


# Main
if __name__ == "__main__":

    # Phase 1 — frozen backbone
    model = build_model(num_classes=len(EMOTIONS), freeze_backbone=True).to(DEVICE)
    # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TUNED)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    h1 = run_training(model, optimizer, scheduler, EPOCHS_FROZEN, "Phase 1 — Frozen Backbone")

    # Phase 2 — fine-tune
    model = unfreeze_backbone(model, unfreeze_from_layer=10)
    # optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TUNED)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TUNED, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    h2 = run_training(model, optimizer, scheduler, EPOCHS_TUNED, "Phase 2 — Fine-tuning")

    plot_history(h1, h2)
    evaluate_test(model)
    print(f"\n✅ Done. Model saved to {SAVE_PATH}")