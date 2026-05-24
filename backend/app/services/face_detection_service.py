from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.base_service import BaseService
from app.utils.preprocess import crop_faces


# det_10g.onnx is InsightFace's SCRFD-10G model.
# It runs detection at 3 scales (strides 8, 16, 32) on a fixed 640x640 input.
# Each stride produces: score per anchor, bbox per anchor, 5 keypoints per anchor.
# We decode all three strides, filter by confidence, then run NMS.

class FaceDetectionService(BaseService):
    def __init__(
        self,
        use_triton: bool | None = None,
        triton_url: str | None = None,
        weights_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        super().__init__(
            "face_detection",
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
            model_path=model_path,
        )
        # Minimum score for a detection to be kept before NMS
        self.confidence_threshold = confidence_threshold
        # IoU threshold for NMS — how much two boxes can overlap before
        # one gets suppressed
        self.nms_threshold = nms_threshold

        # SCRFD operates on a fixed 640x640 input
        self.INPUT_SIZE = (640, 640)

        # SCRFD uses 3 feature map strides.
        # Stride 8  → detects small faces  (fine-grained, many anchors)
        # Stride 16 → detects medium faces
        # Stride 32 → detects large faces  (coarse, fewer anchors)
        self.STRIDES = [8, 16, 32]

        # Number of anchor templates per spatial location per stride.
        # det_10g.onnx uses 2 anchors per location.
        self.NUM_ANCHORS = 2

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Resize to 640x640, normalize to [-1, 1], convert to NCHW float32.

        The model was trained with pixel values in [-1, 1]:
            normalized = (pixel - 127.5) / 128.0

        NCHW means: (Batch, Channels, Height, Width)
        ONNX Runtime expects this layout for image classification/detection models.
        """
        # image arrives as HWC uint8 RGB numpy array from BaseService
        resized = Image.fromarray(image).resize(self.INPUT_SIZE, Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)

        # Normalize: shift [0,255] → [-1, 1]
        arr = (arr - 127.5) / 128.0

        # HWC → CHW → NCHW (add batch dimension)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        input_name = self._input_metadata[0]["name"] if self._input_metadata else "input.1"
        return {input_name: arr.astype(np.float32)}

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(self, outputs: dict[str, np.ndarray]) -> list[dict]:
        """
        Decode raw SCRFD outputs into a flat list of detections.

        det_10g.onnx outputs 9 tensors — 3 per stride:
            stride_8:  score(N,2), bbox(N,8), kps(N,10)
            stride_16: score(N,2), bbox(N,8), kps(N,10)
            stride_32: score(N,2), bbox(N,8), kps(N,10)

        Tensor names follow the pattern:
            score_8,  score_16,  score_32
            bbox_8,   bbox_16,   bbox_32
            kps_8,    kps_16,    kps_32

        But we sort by name and pair them up by stride rather than
        relying on exact names, to be robust across ONNX export versions.

        Returns list of dicts:
            {"bbox_pixels": (x1,y1,x2,y2), "confidence": float}
        All in 640x640 pixel space — detect() converts to normalized coords.
        """
        # Sort output keys so we can group them predictably
        # SCRFD outputs are ordered: all scores, all bboxes, all kps
        # Sorted alphabetically: bbox_16, bbox_32, bbox_8, kps_16, ...
        # We pair them by stride index instead
        output_values = list(outputs.values())

        # Group outputs: SCRFD always outputs in this order per stride —
        # [scores_s8, scores_s16, scores_s32, bboxes_s8, bboxes_s16, bboxes_s32,
        #  kps_s8, kps_s16, kps_s32]
        # 9 outputs total, first 3 are scores, next 3 are bboxes, last 3 are kps
        num_strides = len(self.STRIDES)
        score_outputs = output_values[0:num_strides]              # sigmoid scores
        bbox_outputs  = output_values[num_strides:num_strides*2]  # ltrb deltas
        kps_outputs   = output_values[num_strides*2:]             # 5 keypoints per anchor

        all_detections: list[dict] = []

        for stride, scores, bboxes, kps in zip(
            self.STRIDES, score_outputs, bbox_outputs, kps_outputs
        ):
            detections = self._decode_stride(stride, scores, bboxes, kps)
            all_detections.extend(detections)

        if not all_detections:
            return []

        # Apply NMS across all strides combined
        return self._nms(all_detections)

    def _decode_stride(
        self,
        stride: int,
        scores: np.ndarray,
        bboxes: np.ndarray,
        kps: np.ndarray,
    ) -> list[dict]:
        """
        Decode one stride's score + bbox + keypoint tensors into pixel-space detections.

        scores shape: (1, H*W*num_anchors, 1)   — sigmoid probabilities
        bboxes shape: (1, H*W*num_anchors, 4)   — ltrb distances in stride units
        kps    shape: (1, H*W*num_anchors, 10)  — 5 keypoints as (dx, dy) pairs
                                                   relative to anchor center,
                                                   scaled by stride

        The 5 keypoints in order (InsightFace convention):
            0 — left eye
            1 — right eye
            2 — nose tip
            3 — left mouth corner
            4 — right mouth corner

        Each keypoint is decoded as:
            kp_x = anchor_x + dx * stride
            kp_y = anchor_y + dy * stride

        SCRFD uses the FCOS-style anchor-free representation:
        - Each spatial location has an anchor point at its grid center
        - The bbox prediction is (left, top, right, bottom) distances
          from that anchor point, scaled by the stride
        - So the actual pixel box is:
            x1 = anchor_x - left   * stride
            y1 = anchor_y - top    * stride
            x2 = anchor_x + right  * stride
            y2 = anchor_y + bottom * stride
        """
        # Feature map size at this stride
        feat_h = self.INPUT_SIZE[1] // stride
        feat_w = self.INPUT_SIZE[0] // stride

        # Build anchor center grid — one point per spatial location per anchor
        # Shape: (feat_h * feat_w * num_anchors, 2)
        anchors = self._generate_anchors(feat_h, feat_w, stride)

        # Remove batch dimension, flatten if needed
        scores = scores.reshape(-1)          # (N,)
        bboxes = bboxes.reshape(-1, 4)       # (N, 4) — ltrb
        kps    = kps.reshape(-1, 10)         # (N, 10) — 5 x (dx, dy)

        # Filter by confidence threshold
        keep = scores >= self.confidence_threshold
        if not keep.any():
            return []

        scores  = scores[keep]
        bboxes  = bboxes[keep]
        kps     = kps[keep]
        anchors = anchors[keep]

        # Decode bbox: distances from anchor → pixel x1y1x2y2
        x1 = anchors[:, 0] - bboxes[:, 0] * stride
        y1 = anchors[:, 1] - bboxes[:, 1] * stride
        x2 = anchors[:, 0] + bboxes[:, 2] * stride
        y2 = anchors[:, 1] + bboxes[:, 3] * stride

        # Clip bbox to image bounds (640x640)
        x1 = np.clip(x1, 0, self.INPUT_SIZE[0])
        y1 = np.clip(y1, 0, self.INPUT_SIZE[1])
        x2 = np.clip(x2, 0, self.INPUT_SIZE[0])
        y2 = np.clip(y2, 0, self.INPUT_SIZE[1])

        # Decode keypoints: anchor + offset * stride → pixel (kx, ky)
        # kps columns: [dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4]
        # anchor_x repeated across odd columns, anchor_y across even columns
        kps_decoded = np.zeros_like(kps)   # (N, 10)
        for k in range(5):
            kps_decoded[:, k * 2]     = anchors[:, 0] + kps[:, k * 2]     * stride
            kps_decoded[:, k * 2 + 1] = anchors[:, 1] + kps[:, k * 2 + 1] * stride

        # Clip keypoints to image bounds
        kps_decoded[:, 0::2] = np.clip(kps_decoded[:, 0::2], 0, self.INPUT_SIZE[0])
        kps_decoded[:, 1::2] = np.clip(kps_decoded[:, 1::2], 0, self.INPUT_SIZE[1])

        detections = []
        for i in range(len(scores)):
            # Reshape keypoints into list of 5 (x, y) pairs — still in 640x640 space
            kp_pairs = [
                (float(kps_decoded[i, k * 2]), float(kps_decoded[i, k * 2 + 1]))
                for k in range(5)
            ]
            detections.append({
                "bbox_pixels": (float(x1[i]), float(y1[i]),
                                float(x2[i]), float(y2[i])),
                "confidence":  float(scores[i]),
                "keypoints_pixels": kp_pairs,
                # kp_pairs index meaning:
                # 0 = left eye, 1 = right eye, 2 = nose tip,
                # 3 = left mouth corner, 4 = right mouth corner
            })
        return detections

    def _generate_anchors(
        self, feat_h: int, feat_w: int, stride: int
    ) -> np.ndarray:
        """
        Generate anchor center points for one feature map level.

        For a feature map of size (feat_h, feat_w) with num_anchors=2,
        each spatial location (row, col) gets 2 anchor centers, both
        at the same pixel position (col*stride + stride/2, row*stride + stride/2).

        Returns shape: (feat_h * feat_w * num_anchors, 2) — (cx, cy) in pixels
        """
        # Grid of (col, row) indices
        xs = np.arange(feat_w, dtype=np.float32)
        ys = np.arange(feat_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Center of each grid cell in the 640x640 input space
        cx = (grid_x.ravel() + 0.5) * stride
        cy = (grid_y.ravel() + 0.5) * stride

        # Stack and repeat for num_anchors
        centers = np.stack([cx, cy], axis=1)               # (H*W, 2)
        centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)  # (H*W*2, 2)
        return centers

    def _nms(self, detections: list[dict]) -> list[dict]:
        """
        Non-Maximum Suppression — removes duplicate detections of the same face.

        When multiple overlapping boxes all detect the same face, NMS keeps
        only the highest-confidence one. "Overlap" is measured by IoU
        (Intersection over Union):
            IoU = area of overlap / area of union
        If IoU > nms_threshold, the lower-confidence box is suppressed.
        Keypoints are carried through unchanged — the kept box's keypoints
        are the ones that survive.
        """
        if not detections:
            return []

        boxes  = np.array([d["bbox_pixels"] for d in detections], dtype=np.float32)
        scores = np.array([d["confidence"]  for d in detections], dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score descending
        order = scores.argsort()[::-1]
        kept: list[int] = []

        while order.size > 0:
            i = order[0]
            kept.append(i)

            # Compute IoU of this box against all remaining boxes
            inter_x1 = np.maximum(x1[i], x1[order[1:]])
            inter_y1 = np.maximum(y1[i], y1[order[1:]])
            inter_x2 = np.minimum(x2[i], x2[order[1:]])
            inter_y2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, inter_x2 - inter_x1)
            inter_h = np.maximum(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)

            # Keep boxes with IoU below threshold
            remaining = np.where(iou <= self.nms_threshold)[0]
            order = order[remaining + 1]

        return [detections[i] for i in kept]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image) -> list[dict]:
        """
        Run face detection on a full PIL image.

        Output per face:
          {
            "bbox":       (x, y, w, h),      # normalized [0, 1] — x,y = top-left
            "confidence": float,              # detection score 0.0 – 1.0
            "crop":       PIL.Image.Image,    # face region cropped from original
            "keypoints":  [                   # 5 facial landmarks, normalized [0, 1]
                (x, y),  # 0 — left eye
                (x, y),  # 1 — right eye
                (x, y),  # 2 — nose tip
                (x, y),  # 3 — left mouth corner
                (x, y),  # 4 — right mouth corner
            ]
          }

        Returns an empty list when no faces are found.
        """
        img_w, img_h = image.size

        # Convert to numpy uint8 RGB for BaseService
        img_np = np.asarray(image.convert("RGB"), dtype=np.uint8)

        self._ensure_loaded()

        # preprocess → infer → postprocess
        # postprocess returns pixel-space (x1,y1,x2,y2) and keypoints in 640x640 space
        input_tensors = self.preprocess(img_np)
        raw_outputs   = self._infer(input_tensors)
        detections    = self.postprocess(raw_outputs)

        if not detections:
            return []

        results: list[dict] = []
        bboxes_normalized: list[tuple] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox_pixels"]

            # The model ran on a 640x640 resized image.
            # Scale the pixel coords back to the original image dimensions.
            x1_orig = x1 / self.INPUT_SIZE[0] * img_w
            y1_orig = y1 / self.INPUT_SIZE[1] * img_h
            x2_orig = x2 / self.INPUT_SIZE[0] * img_w
            y2_orig = y2 / self.INPUT_SIZE[1] * img_h

            # Convert (x1,y1,x2,y2) pixel → (x,y,w,h) normalized [0,1]
            # x,y is the top-left corner normalized by image dimensions
            x_norm = x1_orig / img_w
            y_norm = y1_orig / img_h
            w_norm = (x2_orig - x1_orig) / img_w
            h_norm = (y2_orig - y1_orig) / img_h

            # Clamp to [0, 1] — model occasionally predicts slightly outside bounds
            x_norm = float(np.clip(x_norm, 0.0, 1.0))
            y_norm = float(np.clip(y_norm, 0.0, 1.0))
            w_norm = float(np.clip(w_norm, 0.0, 1.0 - x_norm))
            h_norm = float(np.clip(h_norm, 0.0, 1.0 - y_norm))

            bboxes_normalized.append((x_norm, y_norm, w_norm, h_norm))

        # Use the shared crop_faces utility — MUST use this, not custom cropping,
        # so that every downstream service (liveness, emotion, verification)
        # gets crops from exactly the same coordinate math.
        crops = crop_faces(image, bboxes_normalized)

        for bbox, det, crop in zip(bboxes_normalized, detections, crops):
            # Scale keypoints from 640x640 space back to original image,
            # then normalize to [0, 1] by dividing by original dimensions.
            # Follows the same scale-then-normalize logic as bboxes above.
            raw_kps = det["keypoints_pixels"]
            keypoints_normalized = [
                (
                    float(np.clip(kx / self.INPUT_SIZE[0] * img_w / img_w, 0.0, 1.0)),
                    float(np.clip(ky / self.INPUT_SIZE[1] * img_h / img_h, 0.0, 1.0)),
                )
                for kx, ky in raw_kps
            ]

            results.append({
                "bbox":       bbox,
                "confidence": det["confidence"],
                "crop":       crop,
                "keypoints":  keypoints_normalized,
            })

        return results
