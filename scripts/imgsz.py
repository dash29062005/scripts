import torch
from ultralytics import YOLO

# Load the model and print the image size from the model's YAML configuration
model = YOLO(r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt")
# print(model.model.args['imgsz'])
# print(model.model.names)      # classes
# print(model.model.args)       # includes imgsz

model.export(
    format="engine",
    imgsz=1088,
    half=True,
    batch=1,
    workspace=1   # reduce memory
)