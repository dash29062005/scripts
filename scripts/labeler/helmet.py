from ultralytics import YOLO
from huggingface_hub import hf_hub_download

repo_id = "iam-tsr/yolov8n-helmet-detection"
filename = "best.pt"
# Download model
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load downloaded model
model = YOLO(model_path)

# Run inference
result_many = model.predict(
    source=r"D:\gsfc\project\sem8_project\Project_yolo model training\test\D01_20250915_0027525_c0_bike.jpg",
    save=False,
    show=True,
)
