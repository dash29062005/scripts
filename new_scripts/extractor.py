"""
Script 1 — extractor.py
=======================
Samples frames from CCTV videos, runs YOLO, applies quality filters
at capture time, and saves candidate frames + labels to disk.

Run this first. Script 2 (select.py) does the final curation pass.

Quality filters applied here:
  - Confidence threshold
  - Min bounding box pixel size  (removes distant/tiny detections)
  - Min bounding box area        (secondary size gate)
  - Max relative area            (removes objects that fill the whole frame)
"""

import cv2
import os
import time
import json
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────── CONFIG ────────────────────────────────────────
CONFIG = {
    "model_path":    r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt",
    "video_folder":  r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv",

    # Script 1 dumps candidates here; Script 2 reads from here
    "candidate_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\candidates",

    "classes": ["car", "bike", "helmet", "without_helmet", "number_plate"],
    "vehicle_classes": {"car", "bike"},

    # ── Confidence ──────────────────────────────────────────
    "conf_min": 0.35,           # slightly raised — weak detections are noisy

    # ── Bounding-box size gates ─────────────────────────────
    # Absolute pixel thresholds — tune these for your camera resolution.
    # At 1080p, a car 40px wide is basically a speck.
    "min_box_w": 80,            # minimum width  in pixels
    "min_box_h": 60,            # minimum height in pixels
    "min_box_area": 5_000,      # minimum area   in pixels² (w×h)
    "max_box_area_ratio": 0.70, # skip if box occupies > 70% of frame (bad framing)

    # ── Sampling ────────────────────────────────────────────
    "sample_every_n_seconds": 0.2,  # sample twice per second for better coverage
    "device": 0,

    # ── Safety limits ───────────────────────────────────────
    "max_videos": 3,
    "run_time_sec": 240,         # -1 for unlimited
}

CLASSES = CONFIG["classes"]
VEHICLE_CLS = CONFIG["vehicle_classes"]

# ───────────────────────── HELPERS ─────────────────────────────────────────

def setup_folders(base: str):
    img_path = Path(base) / "images"
    lbl_path = Path(base) / "labels"
    img_path.mkdir(parents=True, exist_ok=True)
    lbl_path.mkdir(parents=True, exist_ok=True)
    return str(img_path), str(lbl_path)


def box_to_yolo(x1, y1, x2, y2, frame_w, frame_h):
    cx = ((x1 + x2) / 2) / frame_w
    cy = ((y1 + y2) / 2) / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h
    return cx, cy, bw, bh


def save_label(path: str, detections: list):
    with open(path, "w") as f:
        for cls_id, cx, cy, bw, bh in detections:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def stream_frames(video_path: Path, every_n_seconds: float):
    """Yield (frame_index, frame) at the requested time interval."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * every_n_seconds))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            yield idx, frame
        idx += 1
    cap.release()


def passes_size_filter(x1, y1, x2, y2, frame_w, frame_h) -> bool:
    """Return True only if the bounding box meets all size requirements."""
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    frame_area = frame_w * frame_h

    if bw < CONFIG["min_box_w"]:
        return False
    if bh < CONFIG["min_box_h"]:
        return False
    if area < CONFIG["min_box_area"]:
        return False
    if area / frame_area > CONFIG["max_box_area_ratio"]:
        return False
    return True


# ───────────────────────── MAIN ────────────────────────────────────────────

def main():
    print("Loading YOLO model …")
    model = YOLO(CONFIG["model_path"])

    img_dir, lbl_dir = setup_folders(CONFIG["candidate_folder"])

    videos = sorted(Path(CONFIG["video_folder"]).glob("*.*"))[: CONFIG["max_videos"]]
    if not videos:
        print("No videos found. Check video_folder path.")
        return

    stats = {"car": 0, "bike": 0, "frames_saved": 0, "frames_skipped_size": 0}
    start_time = time.time()

    for vid in videos:
        vid_id = vid.stem
        print(f"\nProcessing: {vid.name}")

        for frame_idx, frame in stream_frames(vid, CONFIG["sample_every_n_seconds"]):

            # ── Time-limit guard ────────────────────────────
            if CONFIG["run_time_sec"] != -1:
                if time.time() - start_time > CONFIG["run_time_sec"]:
                    print("⏹  Time limit reached.")
                    _save_json(stats)
                    return

            frame_h, frame_w = frame.shape[:2]
            results = model(frame, device=CONFIG["device"], verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES[cls_id]

                if cls_name not in VEHICLE_CLS:
                    continue

                conf = float(box.conf[0])
                if conf < CONFIG["conf_min"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Size filter ─────────────────────────────
                if not passes_size_filter(x1, y1, x2, y2, frame_w, frame_h):
                    stats["frames_skipped_size"] += 1
                    continue

                cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, frame_w, frame_h)
                detections.append((cls_id, cx, cy, bw, bh))

                if cls_name == "car":
                    stats["car"] += 1
                elif cls_name == "bike":
                    stats["bike"] += 1

            if detections:
                name = f"{vid_id}_{frame_idx:06d}"
                cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame)
                save_label(os.path.join(lbl_dir, name + ".txt"), detections)
                stats["frames_saved"] += 1

    _save_json(stats)
    print("\n✅ Extraction complete.")
    print(f"   Frames saved  : {stats['frames_saved']}")
    print(f"   Cars detected : {stats['car']}")
    print(f"   Bikes detected: {stats['bike']}")
    print(f"   Skipped (size): {stats['frames_skipped_size']}")


def _save_json(stats: dict):
    out_path = os.path.join(CONFIG["candidate_folder"], "extraction_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved → {out_path}")


if __name__ == "__main__":
    main()
