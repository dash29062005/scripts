import os
import shutil
from ultralytics import YOLO
import torch
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt"

# NEW INPUT (already sorted images)
IMAGES_DIRS = [
    r"D:\gsfc\project\sem8_project\Project_yolo model training\raw\car",
    r"D:\gsfc\project\sem8_project\Project_yolo model training\raw\bike",
    r"D:\gsfc\project\sem8_project\Project_yolo model training\raw\mix"
]

# OUTPUT CROPS
CROP_CAR_DIR = r"D:\gsfc\project\sem8_project\Project_yolo model training\cropped_car"
CROP_BIKE_DIR = r"D:\gsfc\project\sem8_project\Project_yolo model training\cropped_bike"

CONF_THRESHOLD = 0.25
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------- FUNCTIONS ----------------
def load_model():
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    return model


def get_all_images():
    files = []
    for folder in IMAGES_DIRS:
        for f in os.listdir(folder):
            if f.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(folder, f))
    return files


def ensure_unique_filename(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_name = filename

    while os.path.exists(os.path.join(dest_dir, new_name)):
        new_name = f"{base}_{counter}{ext}"
        counter += 1

    return new_name


def crop_and_save(img, box, cls_name, base_name):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return

    if cls_name == "car":
        dest_dir = CROP_CAR_DIR
    elif cls_name == "bike":
        dest_dir = CROP_BIKE_DIR
    else:
        return

    filename = ensure_unique_filename(dest_dir, f"{base_name}.jpg")
    cv2.imwrite(os.path.join(dest_dir, filename), crop)


def run_inference_and_crop(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    results = model(image_path, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)[0]
    names = model.names

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]

        if cls_name in ["car", "bike"]:
            crop_and_save(img, box, cls_name, f"{base_name}_{i}")


def create_output_dirs():
    os.makedirs(CROP_CAR_DIR, exist_ok=True)
    os.makedirs(CROP_BIKE_DIR, exist_ok=True)


# ---------------- MAIN ----------------
def main():
    create_output_dirs()
    model = load_model()
    image_files = get_all_images()

    for img_path in image_files:
        try:
            run_inference_and_crop(model, img_path)
        except:
            continue


if __name__ == "__main__":
    main()