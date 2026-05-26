"""
Train the anti-spoofing MobileNetV2 model and save Keras + ONNX weights under model/.

Run from this folder (or anywhere — paths below are relative to this file):
  python anti_spoofing.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import tensorflow as tf
import tf2onnx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


#============================================================
#General configuration
#============================================================

MODULE_DIR = Path.cwd()

# Expected dataset layout (under MODULE_DIR or repo root if you symlink/copy LCC_FASD):
# LCC_FASD/LCC_FASD_training/, LCC_FASD_development/, LCC_FASD_evaluation/
RAW_DATA_DIR = MODULE_DIR / "LCC_FASD"

MODEL_DIR = MODULE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_STEM = "anti_spoofing"

TRAIN_FOLDER = "LCC_FASD_training"
VAL_FOLDER = "LCC_FASD_development"
TEST_FOLDER = "LCC_FASD_evaluation"

IMAGE_SIZE = 224
IMG_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
IMG_SHAPE = IMG_SIZE + (3,)

BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
FINE_TUNE_AT = 0

BASE_LR = 5e-5
FINE_TUNE_LR = 5e-5
SAVE_ONNX = True
ONNX_OPSET = 13

SEED = 77

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

#============================================================
#Data loading helpers
#============================================================

def get_data_dirs() -> tuple[Path, Path, Path]:
    """Return train, validation, and test directory paths."""
    return (
        RAW_DATA_DIR / TRAIN_FOLDER,
        RAW_DATA_DIR / VAL_FOLDER,
        RAW_DATA_DIR / TEST_FOLDER,
    )


def validate_data_dirs(train_dir: Path, val_dir: Path, test_dir: Path) -> None:
    """Stop the script early if required dataset folders are missing."""
    print("Train dir:", train_dir)
    print("Validation dir:", val_dir)
    print("Test dir:", test_dir)

    print("Train exists:", train_dir.exists())
    print("Validation exists:", val_dir.exists())
    print("Test exists:", test_dir.exists())

    missing_dirs = [
        str(path) for path in [train_dir, val_dir, test_dir]
        if not path.exists()
    ]

    if missing_dirs:
        raise FileNotFoundError(
            "Missing required dataset folders:\n" + "\n".join(missing_dirs)
        )


def load_datasets(train_dir: Path, val_dir: Path, test_dir: Path):
    """Load image datasets from folders."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    return train_ds, val_ds, test_ds


def get_class_counts(dataset) -> dict[str, int]:
    """Count files per class using dataset.file_paths."""
    class_counts = {class_name: 0 for class_name in dataset.class_names}

    for path in dataset.file_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name in class_counts:
            class_counts[class_name] += 1

    return class_counts


def print_and_save_dataset_summary(train_ds, val_ds, test_ds) -> None:
    """Print and save dataset split/class-count summary."""
    train_counts = get_class_counts(train_ds)
    val_counts = get_class_counts(val_ds)
    test_counts = get_class_counts(test_ds)

    summary_rows = []

    for split_name, dataset, class_counts in [
        ("Train", train_ds, train_counts),
        ("Validation", val_ds, val_counts),
        ("Test", test_ds, test_counts),
    ]:
        row = {
            "Split": split_name,
            "Total Files": len(dataset.file_paths),
            "Num Classes": len(dataset.class_names),
        }

        for class_name in dataset.class_names:
            row[f"{class_name} Count"] = class_counts.get(class_name, 0)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = MODEL_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved dataset summary: {summary_path}")


def compute_class_weights(train_ds) -> dict[int, float]:
    """Compute balanced class weights from the training folder counts."""
    train_counts = get_class_counts(train_ds)
    total = sum(train_counts.values())
    num_classes = len(train_counts)

    class_weight = {
        i: total / (num_classes * train_counts[class_name])
        for i, class_name in enumerate(train_ds.class_names)
    }

    print("Class names:", train_ds.class_names)
    print("Training class counts:", train_counts)
    print("Class weights:", class_weight)

    return class_weight


def prepare_datasets(train_ds, val_ds, test_ds):
    """Improve data pipeline performance."""
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


#============================================================
#Model helpers
#============================================================

def build_data_augmentation() -> tf.keras.Sequential:
    """Create data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.10, seed=SEED),
            tf.keras.layers.RandomZoom(0.10, seed=SEED),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),
        ],
        name="data_augmentation",
    )


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build a fresh MobileNetV2 anti-spoofing model."""
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    #1: keep the pretrained base frozen.
    base_model.trainable = False

    data_augmentation = build_data_augmentation()
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    #Extra task-specific convolutional layer for anti-spoofing features.
    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        name="anti_spoof_conv",
    )(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = global_average_layer(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="fasd_mobilenetv2")

    return model, base_model


def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create training callbacks."""
    checkpoint_path = MODEL_DIR / "best_fasd_mobilenetv2_model.keras"

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    return [reduce_lr, early_stop, checkpoint]


#============================================================
#Training and evaluation
#============================================================

def plot_training_history(history, history_fine) -> None:
    """Save training/validation accuracy and loss graph."""
    full_acc = history.history["accuracy"] + history_fine.history["accuracy"]
    full_val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

    full_loss = history.history["loss"] + history_fine.history["loss"]
    full_val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

    fine_tune_start = len(history.history["accuracy"])
    training_graph_path = MODEL_DIR / "training_graph.png"

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(full_acc, label="Training Accuracy")
    plt.plot(full_val_acc, label="Validation Accuracy")
    plt.axvline(fine_tune_start - 1, linestyle="--", label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.ylabel("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(full_loss, label="Training Loss")
    plt.plot(full_val_loss, label="Validation Loss")
    plt.axvline(fine_tune_start - 1, linestyle="--", label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(training_graph_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved training graph: {training_graph_path}")


def evaluate_and_save_outputs(model, train_ds, val_ds, test_ds) -> None:
    """Evaluate model and save metrics, confusion matrix, and classification report."""
    fine_tune_loss, fine_tune_acc = model.evaluate(val_ds)
    print("\nFine-Tuning Validation Results:")
    print("Accuracy: {:.4f}".format(fine_tune_acc))
    print("Loss: {:.4f}".format(fine_tune_loss))

    test_loss, test_acc = model.evaluate(test_ds)
    print("\nTest Results:")
    print("Test Accuracy:", test_acc)
    print("Test Loss:", test_loss)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = train_ds.class_names

    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix_path = MODEL_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {confusion_matrix_path}")

    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(report_text)

    report_path = MODEL_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved classification report: {report_path}")

    metrics = {
        "validation_accuracy": float(fine_tune_acc),
        "validation_loss": float(fine_tune_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")


def save_onnx(model: tf.keras.Model) -> Path:
    """
    Export the trained Keras model to ONNX for backend inference.

    Input shape: (N, 224, 224, 3) float32 RGB in [0, 255] — same as training.
    """
    onnx_path = MODEL_DIR / f"{MODEL_STEM}.onnx"
    input_signature = (
        tf.TensorSpec([None, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32, name="input"),
    )

    @tf.function(input_signature=input_signature)
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    print(f"Converting to ONNX (opset {ONNX_OPSET})...")
    model_proto, _ = tf2onnx.convert.from_function(
        serving_fn,
        input_signature=input_signature,
        opset=ONNX_OPSET,
    )
    onnx.save(model_proto, str(onnx_path))
    print(f"Saved ONNX model: {onnx_path}")
    return onnx_path


def save_final_models(model: tf.keras.Model) -> None:
    """Save Keras weights, and ONNX under model/."""
    keras_model_path = MODEL_DIR / f"{MODEL_STEM}.keras"

    model.save(keras_model_path)
    print(f"Saved Keras model: {keras_model_path}")

    if SAVE_ONNX:
        save_onnx(model)

    best_model_path = MODEL_DIR / "best_fasd_mobilenetv2_model.keras"

    if best_model_path.exists():
        best_model_path.unlink()
        print(f"Deleted temporary checkpoint model: {best_model_path}")
    else:
        print("No temporary checkpoint model found to delete.")


#============================================================
#Main script
#============================================================

def main() -> None:

    train_dir, val_dir, test_dir = get_data_dirs()
    validate_data_dirs(train_dir, val_dir, test_dir)

    train_ds, val_ds, test_ds = load_datasets(train_dir, val_dir, test_dir)
    print_and_save_dataset_summary(train_ds, val_ds, test_ds)

    class_weight = compute_class_weights(train_ds)

    #Save class names before prefetching
    class_names = train_ds.class_names

    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    train_ds.class_names = class_names

    model, base_model = build_model(num_classes=len(class_names))

    print("\nModel summary:")
    model.summary()

    print("\nTraining classifier head with frozen MobileNetV2 base...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    initial_loss, initial_acc = model.evaluate(val_ds)
    print("\nInitial Validation Results:")
    print("Accuracy: {:.4f}".format(initial_acc))
    print("Loss: {:.4f}".format(initial_loss))

    callbacks = create_callbacks()

    history = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    feature_loss, feature_acc = model.evaluate(val_ds)
    print("\nFeature Extraction Validation Results:")
    print("Accuracy: {:.4f}".format(feature_acc))
    print("Loss: {:.4f}".format(feature_loss))

    print("\nFine-tuning MobileNetV2...")
    base_model.trainable = True

    print("Number of layers in the base model:", len(base_model.layers))

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    print("Frozen MobileNetV2 layers:", sum(not layer.trainable for layer in base_model.layers))
    print("Trainable MobileNetV2 layers:", sum(layer.trainable for layer in base_model.layers))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        metrics=["accuracy"],
    )

    feature_epochs_ran = len(history.epoch)
    total_epochs = feature_epochs_ran + FINE_TUNE_EPOCHS

    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=feature_epochs_ran,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    best_model_path = MODEL_DIR / "best_fasd_mobilenetv2_model.keras"

    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)
        print(f"Loaded best model from: {best_model_path}")
    else:
        print("No checkpoint found. Using current model weights.")

    plot_training_history(history, history_fine)

    print("\nStage\t\t\tAccuracy\tLoss")
    print("Initial\t\t\t{:.4f}\t\t{:.4f}".format(initial_acc, initial_loss))
    print("Feature Extraction\t{:.4f}\t\t{:.4f}".format(feature_acc, feature_loss))

    evaluate_and_save_outputs(model, train_ds, val_ds, test_ds)
    save_final_models(model)


if __name__ == "__main__":
    main()
