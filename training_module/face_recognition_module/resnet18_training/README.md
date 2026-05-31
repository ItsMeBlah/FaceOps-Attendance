# ResNet18 ArcFace Trainer

Standalone ResNet18-only training code extracted from `arcface-pytorch`.

The default dataset path is:

```text
dataset/11-785-fall-20-homework-2-part-2/classification_data
```

Expected layout:

```text
classification_data/
  train_data/<class_name>/*.jpg
  val_data/<class_name>/*.jpg
  test_data/<class_name>/*.jpg
```

Run:

```bash
python train_resnet18.py --epochs 50 --batch-size 128 --num-workers 4 --amp
```

Training plots are generated after training finishes:

```text
checkpoints_resnet18/training_curves.png
checkpoints_resnet18/validation_metrics.png
checkpoints_resnet18/test_roc_curve.png
```

Useful options:

```bash
python train_resnet18.py --head arcface --loss focal
python train_resnet18.py --head linear --loss ce
python train_resnet18.py --optimizer sgd --scheduler step --lr 0.1 --lr-step 15 --lr-gamma 0.2
python train_resnet18.py --optimizer adamw --scheduler cosine --lr 0.001 --min-lr 0.000001
python train_resnet18.py --freeze-backbone 3
python train_resnet18.py --image-size 128 --pretrained-backbone weights/resnet18_110.pth
python train_resnet18.py --image-size 128 --resize-size 144
python train_resnet18.py --resume checkpoints_resnet18/last.pth
python train_resnet18.py --verification-negative-ratio 5
python train_resnet18.py --early-stopping-patience 8 --early-stopping-min-delta 0.0005
```

Preprocessing follows the original ArcFace loader: grayscale, train random crop, train horizontal flip, eval center crop, tensor conversion, then normalization with mean/std `0.5`. `--resize-size` runs before the crop; keep it larger than `--image-size` for random crop behavior, or set `--resize-size 0` to disable resizing.

`--freeze-backbone` freezes stages from the input side forward:

```text
0 = no freeze
1 = stem + layer1
2 = stem + layer1 + layer2
3 = stem + layer1 + layer2 + layer3
4 = add layer4
5 = add embedding tail
```

Checkpoints, `training_metadata.json`, and `metrics.csv` are written to `checkpoints_resnet18/` by default. Test confusion matrix outputs are saved as:

```text
checkpoints_resnet18/training_history.json
checkpoints_resnet18/test_confusion_matrix.pt
checkpoints_resnet18/test_confusion_matrix_nonzero.csv
```

Face verification ROC/AUC is enabled by default after training, on the test set only. It evaluates cosine similarity between normalized model embeddings, accepts pairs whose similarity is above a threshold, and logs `test_verification_auc` to `metrics.csv` and `training_history.json`. ROC and pair-score outputs are saved as:

```text
checkpoints_resnet18/test_roc_curve.png
checkpoints_resnet18/test_roc_curve.csv
checkpoints_resnet18/test_verification_pairs.csv
```

Use `--no-verification-eval` to disable it, `--verification-negative-ratio` to control sampled different-person pairs, and `--no-verification-save-pairs` to skip pair CSV files.

Early stopping is disabled by default. Enable it with `--early-stopping-patience`; it monitors validation accuracy and stops after that many epochs without an improvement of at least `--early-stopping-min-delta`. The final test evaluation still runs after early stopping.

## Triplet Loss Training

The triplet training path reuses the same dataset folders, transforms, and ResNet18 backbone. It samples:

```text
anchor   = one face image
positive = another image of the same identity
negative = an image of a different identity
```

Identities with fewer than two images are ignored. The classifier head is not used; the model trains L2-normalized 512-dimensional embeddings with `torch.nn.TripletMarginLoss`.

Run from a classification checkpoint:

```bash
python train_triplet.py \
  --pretrained-backbone checkpoints_resnet18/train_3/best.pth \
  --output-dir checkpoints_resnet18_triplet/train_1 \
  --freeze-backbone 3 \
  --optimizer adamw \
  --scheduler cosine \
  --lr 0.0001 \
  --margin 0.3 \
  --epochs 30
```

Triplet logs include loss, triplet accuracy, mean positive distance, mean negative distance, and active-triplet ratio. Checkpoints use the same `model_state` backbone key as the classification trainer, so they remain compatible with the ONNX exporter.
