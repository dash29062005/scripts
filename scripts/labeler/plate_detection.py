from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import torch
import os
import cv2
import numpy as np

# ---------------- CONFIG ----------------
IMAGE_DIR = r"D:\gsfc\project\sem8_project\Project_yolo model training\raw\crops\images\train"

# ---------------- LOAD MODEL ----------------
feature_extractor = YolosFeatureExtractor.from_pretrained(
    'nickmuchi/yolos-small-finetuned-license-plate-detection'
)
model = YolosForObjectDetection.from_pretrained(
    'nickmuchi/yolos-small-finetuned-license-plate-detection'
)

# ---------------- PROCESS IMAGES ----------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path).convert("RGB")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    # convert PIL -> OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # draw boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_cv, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # resize for display (max 400px)
    h, w = img_cv.shape[:2]
    scale = 400 / max(h, w)
    img_cv = cv2.resize(img_cv, (int(w*scale), int(h*scale)))

    # show
    cv2.imshow("Detection", img_cv)

    key = cv2.waitKey(0)  # press any key for next image
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()