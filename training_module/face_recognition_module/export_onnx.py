from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet18_training.model import resnet18_face


DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent
    / "checkpoints_resnet18"
    / "train_3"
)


class EmbeddingExportWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, normalize_output: bool = True):
        super().__init__()
        self.backbone = backbone
        self.normalize_output = normalize_output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(image)
        if self.normalize_output:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


def load_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "training_metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata.get("config", metadata)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def build_model(config: dict, checkpoint_path: Path, device: torch.device, normalize_output: bool) -> nn.Module:
    input_size = int(config.get("image_size", 128))
    embedding_size = int(config.get("embedding_size", 512))
    use_se = bool(config.get("use_se", False))
    dropout = float(config.get("dropout", 0.0))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    state_dict = strip_module_prefix(state_dict)

    backbone = resnet18_face(
        input_size=input_size,
        embedding_size=embedding_size,
        input_channels=1,
        use_se=use_se,
        dropout=dropout,
    )
    backbone.load_state_dict(state_dict)
    backbone.eval()

    model = EmbeddingExportWrapper(backbone, normalize_output=normalize_output)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained ResNet18 face embedding model to ONNX.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--output", default="")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-dynamic-batch", action="store_true")
    parser.add_argument("--no-normalize-output", action="store_true")
    parser.add_argument("--external-data", action="store_true", help="Store ONNX weights in a separate .data file.")
    parser.add_argument("--verify", action="store_true", help="Verify exported model with onnxruntime if installed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config = load_metadata(checkpoint_dir)
    input_size = int(config.get("image_size", 128))
    output_path = Path(args.output).expanduser().resolve() if args.output else checkpoint_dir / "resnet18_face.onnx"

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        normalize_output=not args.no_normalize_output,
    )

    dummy_input = torch.randn(args.batch_size, 1, input_size, input_size, device=device)
    dynamic_axes = None
    if not args.no_dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        external_data=args.external_data,
        dynamo=False,
    )
    if not args.external_data:
        sidecar_path = output_path.with_name(output_path.name + ".data")
        if sidecar_path.exists():
            sidecar_path.unlink()

    print(f"checkpoint: {checkpoint_path}")
    print(f"onnx: {output_path}")
    print(f"input_shape: [batch, 1, {input_size}, {input_size}]")
    print(f"output_shape: [batch, {int(config.get('embedding_size', 512))}]")
    print(f"normalized_output: {not args.no_normalize_output}")

    if args.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        result = session.run(None, {"input": dummy_input.cpu().numpy()})[0]
        print(f"onnxruntime_output_shape: {list(result.shape)}")
        if not args.no_dynamic_batch:
            second_dummy_input = torch.randn(args.batch_size + 1, 1, input_size, input_size, device=device)
            second_result = session.run(None, {"input": second_dummy_input.cpu().numpy()})[0]
            print(f"onnxruntime_dynamic_batch_check_shape: {list(second_result.shape)}")


if __name__ == "__main__":
    main()
