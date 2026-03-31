import os
import torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt"
IMAGES_DIR = r"D:\gsfc\project\sem8_project\Project_yolo model training\raw"

CONF_THRESHOLD = 0.25

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")


# ---------------- MAIN ----------------
def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    print("CUDA Available:", torch.cuda.is_available())
    print("Using Device:", device)

    model = YOLO(MODEL_PATH)

    # Get just ONE image (dry run)
    image_files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    if not image_files:
        print("No images found")
        return

    test_image = image_files[0]

    try:
        with torch.no_grad():
            results = model.predict(
                source=test_image,
                device=device,
                imgsz=640,
                half=True,
                conf=CONF_THRESHOLD,
                stream=False
            )

        detections = results[0].boxes
        print("Detections:", len(detections))

    except Exception as e:
        print("GPU failed, switching to CPU...")
        with torch.no_grad():
            results = model.predict(
                source=test_image,
                device="cpu",
                imgsz=640,
                conf=CONF_THRESHOLD
            )
        print("CPU run successful")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()