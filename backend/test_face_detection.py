"""
test_face_detection.py
======================
Tests the face detection endpoint in detail.
Checks bbox format, normalization, confidence, crop dimensions,
and multi-face handling.

Run from backend/ root with the server already running:

    python test_face_detection.py --image path/to/face.jpg

Optional multi-face test:
    python test_face_detection.py --image path/to/face.jpg --multi path/to/group.jpg
"""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/api/detection/detect"


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def post_image(image_path: Path) -> requests.Response:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    with image_path.open("rb") as f:
        r = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            files={"file": (image_path.name, f, mime_type)},
            timeout=30,
        )
    return r


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------

def check_status(r: requests.Response) -> bool:
    return check(
        "HTTP 200",
        r.status_code == 200,
        f"got {r.status_code}",
    )


def check_response_shape(data: dict) -> bool:
    has_width  = "image_width"  in data
    has_height = "image_height" in data
    has_faces  = "faces"        in data and isinstance(data["faces"], list)
    ok = has_width and has_height and has_faces
    return check(
        "Response has image_width, image_height, faces",
        ok,
        "" if ok else f"keys present: {list(data.keys())}",
    )


def check_image_dimensions(data: dict) -> bool:
    w = data.get("image_width",  0)
    h = data.get("image_height", 0)
    ok = isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0
    return check(
        "image_width and image_height are positive ints",
        ok,
        f"width={w} height={h}",
    )


def check_at_least_one_face(data: dict) -> bool:
    count = len(data.get("faces", []))
    ok = count >= 1
    return check(
        "At least one face detected",
        ok,
        f"detected {count} face(s)",
    )


def check_bbox_keys(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        has_keys = all(k in bbox for k in ("x", "y", "w", "h"))
        if not has_keys:
            all_ok = False
            print(f"    Face {i}: bbox missing keys — got {list(bbox.keys())}")
    return check(
        "All bboxes have x, y, w, h keys",
        all_ok,
    )


def check_bbox_normalized(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        for key in ("x", "y", "w", "h"):
            val = bbox.get(key, -1)
            if not (0.0 <= val <= 1.0):
                all_ok = False
                print(f"    Face {i}: {key}={val:.6f} is outside [0, 1]")
    return check(
        "All bbox values normalized in [0, 1]",
        all_ok,
    )


def check_bbox_no_overflow(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("w", 0), bbox.get("h", 0)
        if x + w > 1.0 + 1e-5:
            all_ok = False
            print(f"    Face {i}: x + w = {x+w:.6f} exceeds 1.0")
        if y + h > 1.0 + 1e-5:
            all_ok = False
            print(f"    Face {i}: y + h = {y+h:.6f} exceeds 1.0")
    return check(
        "bbox x+w and y+h do not exceed 1.0",
        all_ok,
    )


def check_confidence(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        conf = face.get("detection_confidence", -1)
        if not (0.0 <= conf <= 1.0):
            all_ok = False
            print(f"    Face {i}: confidence={conf} is outside [0, 1]")
    return check(
        "All confidence values in [0, 1]",
        all_ok,
    )


def check_confidence_type(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        conf = face.get("detection_confidence")
        if not isinstance(conf, float):
            all_ok = False
            print(f"    Face {i}: confidence type is {type(conf).__name__}, expected float")
    return check(
        "Confidence values are plain Python floats (not numpy scalars)",
        all_ok,
    )


def check_crop_dimensions(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        cw = face.get("crop_width",  0)
        ch = face.get("crop_height", 0)
        if cw <= 0 or ch <= 0:
            all_ok = False
            print(f"    Face {i}: crop_width={cw} crop_height={ch}")
    return check(
        "All crops have positive width and height",
        all_ok,
    )


def check_keypoints_present(faces: list) -> bool:
    all_ok = True
    for i, face in enumerate(faces):
        kps = face.get("keypoints")
        if kps is None:
            all_ok = False
            print(f"    Face {i}: 'keypoints' key missing from response")
        elif not isinstance(kps, list):
            all_ok = False
            print(f"    Face {i}: keypoints is {type(kps).__name__}, expected list")
        elif len(kps) != 5:
            all_ok = False
            print(f"    Face {i}: expected 5 keypoints, got {len(kps)}")
    return check(
        "All faces have keypoints list of length 5",
        all_ok,
    )


def check_keypoints_normalized(faces: list) -> bool:
    NAMES = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]
    all_ok = True
    for i, face in enumerate(faces):
        kps = face.get("keypoints", [])
        for k, point in enumerate(kps):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                all_ok = False
                print(f"    Face {i} kp[{k}]: expected (x, y) pair, got {point}")
                continue
            x, y = point
            if not (0.0 <= x <= 1.0):
                all_ok = False
                print(f"    Face {i} kp[{k}] {NAMES[k]}: x={x:.6f} outside [0, 1]")
            if not (0.0 <= y <= 1.0):
                all_ok = False
                print(f"    Face {i} kp[{k}] {NAMES[k]}: y={y:.6f} outside [0, 1]")
    return check(
        "All keypoint coordinates normalized in [0, 1]",
        all_ok,
    )


def check_keypoints_inside_bbox(faces: list) -> bool:
    """
    Landmarks should roughly sit inside or very near the face bbox.
    Uses a small tolerance since keypoints near the face boundary
    can legitimately fall slightly outside.
    """
    NAMES = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]
    TOLERANCE = 0.05
    all_ok = True
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("w", 0), bbox.get("h", 0)
        kps = face.get("keypoints", [])
        for k, point in enumerate(kps):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            kx, ky = point
            in_x = (x - TOLERANCE) <= kx <= (x + w + TOLERANCE)
            in_y = (y - TOLERANCE) <= ky <= (y + h + TOLERANCE)
            if not (in_x and in_y):
                all_ok = False
                print(
                    f"    Face {i} kp[{k}] {NAMES[k]}: "
                    f"({kx:.4f}, {ky:.4f}) far outside "
                    f"bbox ({x:.4f}, {y:.4f}, {w:.4f}, {h:.4f})"
                )
    return check(
        "All keypoints within bbox bounds (±0.05 tolerance)",
        all_ok,
    )


def check_empty_on_no_face() -> bool:
    """
    Sends a plain white 100x100 image — should return 0 faces, not crash.
    """
    separator("Edge case — blank image (expect 0 faces, not a crash)")
    import io
    from PIL import Image as PILImage

    blank = PILImage.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="JPEG")
    buf.seek(0)

    r = requests.post(
        f"{BASE_URL}{ENDPOINT}",
        files={"file": ("blank.jpg", buf, "image/jpeg")},
        timeout=30,
    )

    data = {}
    try:
        data = r.json()
    except Exception:
        pass

    ok_status = r.status_code == 200
    ok_empty  = isinstance(data.get("faces"), list) and len(data["faces"]) == 0
    ok_not_none = data.get("faces") is not None

    check("Returns HTTP 200 on blank image",     ok_status, f"got {r.status_code}")
    check("faces is [] not None on blank image", ok_not_none and ok_empty,
          f"faces={data.get('faces')}")

    return ok_status and ok_empty


def check_invalid_file() -> bool:
    """
    Sends a text file — should return 400, not 500.
    """
    separator("Edge case — non-image file (expect 400)")
    r = requests.post(
        f"{BASE_URL}{ENDPOINT}",
        files={"file": ("bad.txt", b"this is not an image", "text/plain")},
        timeout=30,
    )
    ok = r.status_code == 400
    check("Returns 400 for non-image upload", ok, f"got {r.status_code}")
    return ok


def check_multi_face(image_path: Path) -> bool:
    """
    Verifies multiple faces are all returned, each with valid format.
    """
    separator(f"Multi-face image — {image_path.name}")
    r = post_image(image_path)
    if r.status_code != 200:
        check("HTTP 200", False, f"got {r.status_code}")
        return False

    data = r.json()
    faces = data.get("faces", [])
    count = len(faces)
    print(f"  Faces detected: {count}")

    results = [
        check_bbox_keys(faces),
        check_bbox_normalized(faces),
        check_bbox_no_overflow(faces),
        check_confidence(faces),
        check_crop_dimensions(faces),
        check_keypoints_present(faces),
        check_keypoints_normalized(faces),
        check_keypoints_inside_bbox(faces),
    ]

    if count > 1:
        # Check no two bboxes are identical (NMS should have deduplicated)
        bboxes = [
            (f["bbox"]["x"], f["bbox"]["y"], f["bbox"]["w"], f["bbox"]["h"])
            for f in faces
        ]
        unique = len(set(bboxes)) == len(bboxes)
        results.append(check("No duplicate bboxes after NMS", unique))

    return all(results)


# ------------------------------------------------------------------
# Pretty-print all faces
# ------------------------------------------------------------------

def print_faces(data: dict) -> None:
    NAMES = ["L.eye", "R.eye", "Nose", "L.mouth", "R.mouth"]
    separator("Detected faces detail")
    faces = data.get("faces", [])
    if not faces:
        print("  No faces.")
        return
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        print(
            f"  Face {i+1}:"
            f"  x={bbox['x']:.4f}  y={bbox['y']:.4f}"
            f"  w={bbox['w']:.4f}  h={bbox['h']:.4f}"
            f"  conf={face['detection_confidence']:.4f}"
            f"  crop=({face['crop_width']}x{face['crop_height']})"
        )
        kps = face.get("keypoints", [])
        if kps:
            kp_str = "  Keypoints: " + "  ".join(
                f"{NAMES[k]}=({kps[k][0]:.4f},{kps[k][1]:.4f})"
                for k in range(len(kps))
            )
            print(f"  {kp_str}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    global BASE_URL
    parser = argparse.ArgumentParser(description="Test face detection endpoint")
    parser.add_argument("--image", required=True,  help="Path to a face image")
    parser.add_argument("--multi",                 help="Path to a multi-face image (optional)")
    parser.add_argument("--url",   default=BASE_URL, help="Base server URL")
    args = parser.parse_args()

    BASE_URL = args.url.rstrip("/")

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    print(f"\nFace Detection Tests — {BASE_URL}{ENDPOINT}")
    print(f"Image: {image_path.resolve()}")

    # --- Main image tests ---
    separator("Single face image — basic checks")
    r = post_image(image_path)
    print(f"Raw response status: {r.status_code}")

    data = {}
    try:
        data = r.json()
        print(f"Raw response:\n{json.dumps(data, indent=2)}")
    except Exception:
        print(f"Could not parse JSON: {r.text}")

    faces = data.get("faces", [])

    results = {
        "http_200":                check_status(r),
        "response_shape":          check_response_shape(data),
        "image_dimensions":        check_image_dimensions(data),
        "at_least_one_face":       check_at_least_one_face(data),
        "bbox_keys":               check_bbox_keys(faces),
        "bbox_normalized":         check_bbox_normalized(faces),
        "bbox_no_overflow":        check_bbox_no_overflow(faces),
        "confidence_range":        check_confidence(faces),
        "confidence_type":         check_confidence_type(faces),
        "crop_dimensions":         check_crop_dimensions(faces),
        "keypoints_present":       check_keypoints_present(faces),
        "keypoints_normalized":    check_keypoints_normalized(faces),
        "keypoints_inside_bbox":   check_keypoints_inside_bbox(faces),
        "blank_image":             check_empty_on_no_face(),
        "invalid_file":            check_invalid_file(),
    }

    print_faces(data)

    # --- Optional multi-face ---
    if args.multi:
        multi_path = Path(args.multi)
        if multi_path.is_file():
            results["multi_face"] = check_multi_face(multi_path)
        else:
            print(f"\nMulti-face image not found: {multi_path}")

    # --- Summary ---
    separator("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
        if not passed:
            all_passed = False

    print(f"\n{'All tests passed.' if all_passed else 'Some tests failed — see above.'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
