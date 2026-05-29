import json
import os
from pathlib import Path
from PIL import Image
import random

screenshots_dir = Path(
    os.environ.get(
        "CAPTURES_DIR",
        "backend/captures/screenshots"
    )
)

train_fasd_path = Path(os.environ["TRAIN_FASD_PATH"])
test_fasd_path = Path(os.environ["TEST_FASD_PATH"])

train_affectnet_path = Path(os.environ["TRAIN_AFFECTNET_PATH"])
test_affectnet_path = Path(os.environ["TEST_AFFECTNET_PATH"])

recognition_path = Path(os.environ["RECOGNITION_PATH"])

#Confidence Thresholds
min_detection_confidence = 0.7
min_label_confidence = 0.7


#Read all paths
def read_path(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def get_face(data, number):
    face_number = data["faces"][number]
    
    return face_number

def get_properties(data, face_number):
    image_width = data["image_width"]
    image_height = data["image_height"]

    bbox = face_number["face"]["bbox"]
    detection_confidence = face_number["face"]["detection_confidence"]

    emotion = face_number["emotion"]
    anti_spoofing = face_number["anti_spoofing"]
    recognition = face_number["recognition"]

    return image_width, image_height, bbox, detection_confidence, emotion, anti_spoofing, recognition


def build_bounding_box(bbox, image_width, image_height):
    left = bbox["x"] * image_width
    top = bbox["y"] * image_height
    right = (bbox["x"] + bbox["w"]) * image_width
    bottom = (bbox["y"] + bbox["h"]) * image_height

    return int(left), int(top), int(right), int(bottom)

def crop_face(image, bbox, image_width, image_height):
    left, top, right, bottom = build_bounding_box(bbox, image_width, image_height)

    return image.crop((left, top, right, bottom))


def save_split():
    if random.random() < 0.7:
        value = 0
    else:
        value = 1
    return value

def save_recognition_split():
    value = random.random()

    if value < 0.33:
        return "train"
    elif value < 0.67:
        return "val"
    else:
        return "test"


def main():
    for file in screenshots_dir.glob("*.json"):
        data = read_path(file)

        image_file = file.with_suffix(".jpg")

        saved_true = False

        for i in range(len(data["faces"])):
            face = get_face(data, i)

            image_width, image_height, bbox, detection_confidence, emotion, anti_spoofing, recognition = get_properties(data, face)

            if detection_confidence < min_detection_confidence:
                print(f"Skipped {file.name} Face {i + 1}: low detection confidence")
                continue

            image = Image.open(image_file)
            cropped_face = crop_face(image, bbox, image_width, image_height)

            if anti_spoofing["confidence"] >= min_label_confidence:
                spoofing_label = anti_spoofing["label"]

                split = save_split()
                if split == 0:
                    output_path = train_fasd_path / spoofing_label
                else:
                    output_path = test_fasd_path / spoofing_label

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(f"Saved {file.name} Face {i + 1} to {output_file} Confidence: {anti_spoofing['confidence']}")

            if emotion["confidence"] >= min_label_confidence:
                emotion_label = emotion["label"]

                split = save_split()
                if split == 0:
                    output_path = train_affectnet_path / emotion_label
                else:
                    output_path = test_affectnet_path / emotion_label

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(f"Saved {file.name} Face {i + 1} to {output_file} Confidence: {emotion['confidence']}")

            if recognition["confidence"] >= min_label_confidence:
                if recognition["label"] == "unknown" or recognition["matched"] is False:
                    continue

                identity = recognition["label"]

                split = save_recognition_split()

                split_dirs = {
                    "train": recognition_path / "train_data",
                    "val": recognition_path / "val_data",
                    "test": recognition_path / "test_data",
                }

                # Ensure class folders exist in ALL splits
                for split_dir in split_dirs.values():
                    (split_dir / identity).mkdir(parents=True, exist_ok=True)

                output_path = split_dirs[split] / identity

                output_file = output_path / f"{file.stem}_{i + 1}.jpg"

                cropped_face.save(output_file)

                saved_true = True

                print(
                    f"Saved {file.name} Face {i + 1} "
                    f"to {output_file} "
                    f"Confidence: {recognition['confidence']}"
                )
    
        if saved_true:
            file.unlink()
            image_file.unlink()

            print(f"Deleted {file.name} and {image_file.name}")

if __name__ == "__main__":
    main()