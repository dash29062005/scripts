from ultralytics import YOLO

model = YOLO(r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt")

model.export(
    format="engine",
    imgsz=1088,    # matches your training exactly
    device=0,      # GTX 1650
    half=True,     # FP16 — fits in 4GB VRAM, ~2x faster
)