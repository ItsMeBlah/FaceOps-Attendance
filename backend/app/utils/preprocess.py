from __future__ import annotations

import io

import numpy as np
from PIL import Image
import cv2
from skimage import transform as trans


src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112, mode: str = "arcface") -> np.ndarray:
	landmark = np.asarray(landmark, dtype=np.float32)
	M, _ = estimate_norm(landmark, image_size, mode)
	return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


def align_image_by_eyes(
	img: np.ndarray,
	landmark: np.ndarray,
	image_size: int = 112,
	mode: str = "arcface",
	border_value: tuple[int, int, int] | int = 0,
) -> np.ndarray:
	"""
	Align the full image using the same landmark transform as norm_crop().

	Unlike norm_crop(), this keeps only the rotation component and preserves
	the original image canvas, so it does not crop the face into a template.
	landmark must contain 5 pixel-space points in this order:
	left eye, right eye, nose, left mouth, right mouth.
	"""
	landmark = np.asarray(landmark, dtype=np.float32)
	if landmark.shape != (5, 2):
		raise ValueError("landmark must have shape (5, 2).")

	norm_matrix, _ = estimate_norm(landmark, image_size=image_size, mode=mode)
	rotation = norm_matrix[:, :2].astype(np.float32)
	scale = float(np.linalg.norm(rotation[:, 0]))
	if scale <= 0.0:
		raise ValueError("Unable to estimate a valid face alignment rotation.")

	rotation = rotation / scale
	center = landmark[:2].mean(axis=0)
	translation = center - rotation @ center
	align_matrix = np.column_stack([rotation, translation]).astype(np.float32)

	img_h, img_w = img.shape[:2]
	return cv2.warpAffine(
		img,
		align_matrix,
		(img_w, img_h),
		flags=cv2.INTER_LINEAR,
		borderValue=border_value,
	)


def blur_score(image: Image.Image | np.ndarray) -> float:
	"""Measure image sharpness with the variance of the Laplacian."""
	if isinstance(image, Image.Image):
		arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
	else:
		arr = np.asarray(image)

	if arr.ndim == 3:
		gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
	elif arr.ndim == 2:
		gray = arr
	else:
		raise ValueError("image must be a 2D grayscale or 3D RGB array.")

	return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_image_from_bytes(contents: bytes) -> Image.Image:
	image = Image.open(io.BytesIO(contents))
	return image.convert("RGB")

def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
	"""Crop a single normalised (x, y, w, h) box from a PIL image."""
	x, y, w, h = box
	img_w, img_h = image.size
	x1 = int(max(0, x * img_w))
	y1 = int(max(0, y * img_h))
	x2 = int(min(img_w, (x + w) * img_w))
	y2 = int(min(img_h, (y + h) * img_h))
	x2 = max(x2, x1 + 1)
	y2 = max(y2, y1 + 1)
	return image.crop((x1, y1, x2, y2))


def crop_faces(
	image: Image.Image,
	bboxes: list[tuple[float, float, float, float]],
	resize: tuple[int, int] | None = None,
) -> list[Image.Image]:
	"""
	Crop all bboxes from image in one pass.

	Parameters
	----------
	image   : PIL.Image.Image
	bboxes  : list of (x, y, w, h) in normalised [0, 1] coordinates
	resize  : (width, height) to resize each crop; None = no resize

	Returns
	-------
	list[PIL.Image.Image]  — one crop per bbox, same order
	"""
	img_w, img_h = image.size
	img_np = np.array(image)
	crops: list[Image.Image] = []

	for x, y, w, h in bboxes:
		x1 = int(max(0, x * img_w))
		y1 = int(max(0, y * img_h))
		x2 = int(min(img_w, (x + w) * img_w))
		y2 = int(min(img_h, (y + h) * img_h))
		crop = Image.fromarray(img_np[y1:y2, x1:x2])
		if resize is not None:
			crop = crop.resize(resize, Image.BILINEAR)
		crops.append(crop)

	return crops

def crop_faces_v2(
	image: Image.Image,
	keypoints: list[list[tuple[float, float]]],
	image_size: int = 128,
	mode: str = "arcface",
) -> list[Image.Image]:
	"""
	Align face crops from 5 facial keypoints.

	Parameters
	----------
	image      : PIL.Image.Image
	keypoints  : list of 5-point landmarks per face, normalized [0, 1]
	             order: left eye, right eye, nose, left mouth, right mouth
	image_size : output aligned face size, default 128
	mode       : alignment template mode, default "arcface"

	Returns
	-------
	list[PIL.Image.Image] — one aligned crop per face, same order
	"""
	img_w, img_h = image.size
	img_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
	crops: list[Image.Image] = []

	for face_keypoints in keypoints:
		landmark = np.asarray(face_keypoints, dtype=np.float32)
		if landmark.shape != (5, 2):
			raise ValueError("Each face must have 5 keypoints with shape (5, 2).")
		landmark[:, 0] *= img_w
		landmark[:, 1] *= img_h

		aligned = norm_crop(img_np, landmark, image_size=image_size, mode=mode)
		crops.append(Image.fromarray(aligned.astype(np.uint8)))

	return crops
