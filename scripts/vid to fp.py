import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------- CONFIG ----------------
video_list = [
    (1, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv\D01_20250915092037.mp4"),
    (2, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv\D01_20250915092423.mp4"),
    (3, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv\D01_20250915095900.mp4"),
    (4, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv2\D02_20250915091522.avi"),
    (5, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv1\D02_20250915162959.avi"),
    (6, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv1\D02_20250915164419.avi"),
    (7, r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv1\D02_20250915171843.avi")
]

output_dir = r"D:\gsfc\project\sem8_project\Project_yolo model training\raw"
os.makedirs(output_dir, exist_ok=True)

interval_sec = 0.75
MAX_WORKERS = 2   # safe for 8GB RAM

# ---------------- GLOBAL ----------------
counter_lock = threading.Lock()
global_count = 0

# ---------------- FUNCTION ----------------
def process_video(vid_id, video_path):
    global global_count

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Skipping {video_path} (FPS not detected)")
        return 0

    frame_interval = max(1, int(fps * interval_sec))
    frame_count = 0
    saved = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            with counter_lock:
                unique_id = global_count
                global_count += 1

            filename = f"{vid_id}_{unique_id:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1

        frame_count += 1

    cap.release()
    print(f"{video_name} → {saved} frames")
    return saved

# ---------------- PARALLEL EXECUTION ----------------
total_saved = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_video, vid, path) for vid, path in video_list]

    for f in as_completed(futures):
        total_saved += f.result()

print(f"\nDone: Extracted {total_saved} frames from all videos.")