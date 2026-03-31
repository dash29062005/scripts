from ultralytics import YOLO
import cv2

model = YOLO(r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.engine")

# test on one frame
frame = cv2.imread(r"D:\gsfc\project\sem8_project\Project_yolo model training\test\D02_20250915_0008225.jpg")
results = model(frame, imgsz=1088, half=True)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        label = model.names[cls_id]
        print(f"{label}: {conf:.2f}  [{x1},{y1},{x2},{y2}]")

    cv2.imshow("TRT", r.plot())
    cv2.waitKey(0)