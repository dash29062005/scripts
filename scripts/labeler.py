"""
=============================================================
  YOLOv8 Semi-Auto Dataset Generator  (Two-Stage Edition)
  For: Windows PC | Classes: car, bike, helmet, without_helmet, number_plate
=============================================================
  TWO-STAGE DETECTION FLOW:
    Stage 1 → Full frame  → detect car / bike only
    Stage 2 → Vehicle ROI → detect helmet / without_helmet / number_plate
              coordinates mapped back to full frame
              merged into single label file

  KEYBOARD SHORTCUTS (Review Window):
    A = Accept detection as-is
    S = Skip / discard this frame
    E = Edit bounding box (click and drag to redraw)
    N = Next image in batch
    Q = Quit review early (progress saved)
=============================================================
"""

import cv2
import numpy as np
import os
import sys
import json
import random
from pathlib import Path

# ── Try importing ultralytics ──────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] ultralytics not installed.")
    print("  Run:  pip install ultralytics\n")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  CONFIG  ← Edit these paths before running
# ══════════════════════════════════════════════════════════
CONFIG = {
    # Path to your .pt model file
    "model_path": r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt",

    # Folder containing your video files
    "video_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv",

    # Where the dataset will be saved
    "output_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\raw",

    # Your 5 class names (must match model order exactly)
    "classes": ["car", "bike", "helmet", "without_helmet", "number_plate"],

    # Confidence thresholds
    "conf_auto_accept": 0.50,  # above this  → auto accepted, no review needed
    "conf_discard":     0.20,  # below this  → thrown away silently

    # Stage 2 confidence (lower threshold — small objects score lower)
    "conf_stage2_min":  0.20,  # minimum conf to accept a plate/helmet from crop

    # Frame sampling
    "sample_every_n_seconds": 2,  # 1 frame every 2 seconds

    # Minimum vehicle box size in full frame (skip far-away vehicles)
    "min_vehicle_width":  80,   # pixels
    "min_vehicle_height": 80,   # pixels

    # Crop padding around vehicle ROI before Stage 2
    "crop_padding": 20,  # pixels

    # Train / val split
    "val_split": 0.15,  # 15% val, 85% train

    # Resize 4K frames before YOLO (prevents RAM/OOM crash)
    # 1280 = good balance of speed vs accuracy for GTX 1650
    # lower = faster but may miss small objects
    "resize_max_dim": 960,

    # GPU device: 0 = use your GTX 1650, "cpu" = CPU only
    "device": 0,
}
# ══════════════════════════════════════════════════════════

CLASSES     = CONFIG["classes"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
VEHICLE_CLS = {"car", "bike"}
SMALL_CLS   = {"helmet", "without_helmet", "number_plate"}

# ── Resize frame (keeps aspect ratio) ─────────────────────
def resize_frame(frame, max_dim):
    """Resize so longest side = max_dim. Fixes 4K OOM errors."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


# ── Output folder setup ────────────────────────────────────
def setup_folders(base: str) -> dict:
    folders = {
        "full_images_train": Path(base) / "images" / "train",
        "full_images_val":   Path(base) / "images" / "val",
        "full_labels_train": Path(base) / "labels" / "train",
        "full_labels_val":   Path(base) / "labels" / "val",
        "crop_images_train": Path(base) / "crops" / "images" / "train",
        "crop_images_val":   Path(base) / "crops" / "images" / "val",
        "crop_labels_train": Path(base) / "crops" / "labels" / "train",
        "crop_labels_val":   Path(base) / "crops" / "labels" / "val",
        "review_queue":      Path(base) / "_review_queue",
        "session":           Path(base) / "_session",
    }
    for f in folders.values():
        f.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in folders.items()}


# ── Blur detection ─────────────────────────────────────────
def is_blurry(img_crop: np.ndarray, threshold: float = 80.0) -> bool:
    if img_crop is None or img_crop.size == 0:
        return True
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


# ── YOLO label format ──────────────────────────────────────
def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def yolo_to_box(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def save_label(label_path: str, detections: list):
    """detections = list of (class_id, cx, cy, w, h)"""
    with open(label_path, "w") as f:
        for d in detections:
            f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f}\n")


# ── Session state (resume support) ────────────────────────
def load_session(session_dir: str) -> dict:
    path = os.path.join(session_dir, "session.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed_videos": [], "stats": {"auto_accepted": 0, "reviewed": 0, "discarded": 0}}


def save_session(session_dir: str, state: dict):
    path = os.path.join(session_dir, "session.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ── Frame extractor (streaming — does NOT load all frames into RAM) ──
def stream_frames(video_path: str, every_n_seconds: int):
    """Generator: yields (frame_idx, frame) one at a time. Safe for 4K long videos."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * every_n_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            yield frame_idx, frame, fps, total_frames
        frame_idx += 1
    cap.release()


# ── Simple centroid tracker ────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}       # id → centroid
        self.disappeared = {}   # id → count
        self.max_disappeared = max_disappeared
        self.track_history = {} # id → list of frame indices

    def register(self, centroid, frame_idx):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.track_history[self.next_id] = [frame_idx]
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, centroids, frame_idx):
        if len(centroids) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {}

        if len(self.objects) == 0:
            for c in centroids:
                self.register(c, frame_idx)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            assigned = {}
            used_input = set()

            for i, obj_id in enumerate(obj_ids):
                best_dist = float("inf")
                best_j = -1
                for j, c in enumerate(centroids):
                    if j in used_input:
                        continue
                    dist = np.linalg.norm(np.array(obj_centroids[i]) - np.array(c))
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_dist < 150 and best_j >= 0:
                    assigned[obj_id] = best_j
                    used_input.add(best_j)

            for obj_id, j in assigned.items():
                self.objects[obj_id] = centroids[j]
                self.disappeared[obj_id] = 0
                self.track_history[obj_id].append(frame_idx)

            for obj_id in obj_ids:
                if obj_id not in assigned:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

            for j, c in enumerate(centroids):
                if j not in used_input:
                    self.register(c, frame_idx)

        return dict(self.objects)


# ── Smart frame selector per track ────────────────────────
def select_track_frames(frame_indices: list) -> list:
    """Pick first, mid, last — max 3 frames per track."""
    if len(frame_indices) == 0:
        return []
    if len(frame_indices) <= 3:
        return frame_indices
    mid = frame_indices[len(frame_indices) // 2]
    return [frame_indices[0], mid, frame_indices[-1]]


# ══════════════════════════════════════════════════════════
#  REVIEW UI
# ══════════════════════════════════════════════════════════
class ReviewUI:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rect = None
        self.edit_mode = False
        self.current_display = None
        self.edit_class_id = 0

    def draw_callback(self, event, x, y, flags, param):
        if not self.edit_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.current_display.copy()
            cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("REVIEW", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect = (min(self.ix, x), min(self.iy, y),
                         max(self.ix, x), max(self.iy, y))

    def review_batch(self, batch: list) -> list:
        """
        batch = list of dicts:
          { "frame": np.array, "detections": [(cls_id, cx, cy, w, h, conf)],
            "img_h": int, "img_w": int }
        Returns list of accepted detections per item.
        """
        results = []
        cv2.namedWindow("REVIEW", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("REVIEW", 1280, 720)
        cv2.setMouseCallback("REVIEW", self.draw_callback)

        for i, item in enumerate(batch):
            frame    = item["frame"].copy()
            dets     = item["detections"]   # [(cls_id, cx, cy, w, h, conf)]
            img_h, img_w = item["img_h"], item["img_w"]
            accepted = list(dets)           # start with all detections
            self.edit_mode = False
            self.rect = None

            # Draw all boxes — green for vehicles, orange for sub-objects
            display = frame.copy()
            for d in dets:
                cls_id, cx, cy, w, h, conf = d
                x1, y1, x2, y2 = yolo_to_box(cx, cy, w, h, img_w, img_h)
                cls_name = CLASSES[cls_id]
                if cls_name in VEHICLE_CLS:
                    color = (0, 200, 80)    # green  = vehicle (Stage 1)
                else:
                    color = (0, 140, 255)   # orange = helmet/plate (Stage 2)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(display, label, (x1, max(y1-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # HUD
            info = f"  [{i+1}/{len(batch)}]  A=Accept  S=Skip  E=Edit  N=Next  Q=Quit"
            cv2.putText(display, info, (10, img_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            self.current_display = display
            cv2.imshow("REVIEW", display)

            while True:
                key = cv2.waitKey(20) & 0xFF

                if key == ord('a'):   # Accept
                    results.append({"accepted": True, "detections": accepted})
                    break

                elif key == ord('s'): # Skip
                    results.append({"accepted": False, "detections": []})
                    break

                elif key == ord('n'): # Next (accept and move on)
                    results.append({"accepted": True, "detections": accepted})
                    break

                elif key == ord('q'): # Quit review early
                    cv2.destroyAllWindows()
                    return results

                elif key == ord('e'): # Edit mode
                    self.edit_mode = True
                    self.rect = None
                    hint = display.copy()
                    cv2.putText(hint, "Draw new box, then press A to confirm",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.imshow("REVIEW", hint)

                    # Wait for box to be drawn then confirmed
                    while True:
                        k2 = cv2.waitKey(20) & 0xFF
                        if k2 == ord('a') and self.rect is not None:
                            x1, y1, x2, y2 = self.rect
                            # Use first detection's class, or 0
                            cls_id = dets[0][0] if dets else 0
                            cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, img_w, img_h)
                            accepted = [(cls_id, cx, cy, bw, bh, 1.0)]
                            self.edit_mode = False
                            results.append({"accepted": True, "detections": accepted})
                            break
                        elif k2 == ord('s'):
                            self.edit_mode = False
                            results.append({"accepted": False, "detections": []})
                            break
                    break

        cv2.destroyAllWindows()
        return results


# ══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline():
    print("\n" + "="*60)
    print("  YOLOv8 Semi-Auto Dataset Generator")
    print("="*60)

    # ── Validate config ──
    if not os.path.exists(CONFIG["model_path"]):
        print(f"\n[ERROR] Model not found: {CONFIG['model_path']}")
        print("  Please edit CONFIG['model_path'] in this script.\n")
        sys.exit(1)

    if not os.path.isdir(CONFIG["video_folder"]):
        print(f"\n[ERROR] Video folder not found: {CONFIG['video_folder']}")
        print("  Please edit CONFIG['video_folder'] in this script.\n")
        sys.exit(1)

    # ── Setup ──
    folders = setup_folders(CONFIG["output_folder"])
    session = load_session(folders["session"])
    model   = YOLO(CONFIG["model_path"])
    model.to(f'cuda:{CONFIG["device"]}' if isinstance(CONFIG["device"], int) else CONFIG["device"])
    print(f'  Model loaded on: {next(model.model.parameters()).device}')
    ui      = ReviewUI()

    stats   = session["stats"]

    # ── Video list ──
    exts = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4"}
    videos = [
        f for f in Path(CONFIG["video_folder"]).iterdir()
        if f.suffix in exts and f.name not in session["processed_videos"]
    ]

    if not videos:
        print("\n[INFO] No new videos to process. All done!")
        return

    print(f"\n  Found {len(videos)} video(s) to process.")
    print(f"  Output → {CONFIG['output_folder']}\n")

    # ── Process each video ──
    for vid_path in videos:
        vid_id = vid_path.stem[:12].replace(" ", "_")
        print(f"\n{'─'*50}")
        print(f"  Processing: {vid_path.name}")
        print(f"{'─'*50}")

        tracker     = CentroidTracker(max_disappeared=8)
        frame_store = {}   # frame_idx → resized frame
        review_q    = []   # frames that need human review
        max_dim     = CONFIG["resize_max_dim"]
        device      = CONFIG["device"]

        # ── Stage 1 pass: stream + resize + detect vehicles ──
        print("  Stage 1: streaming video, resizing frames, tracking vehicles...")
        frame_count = 0
        for frame_idx, frame, fps, total in stream_frames(str(vid_path), CONFIG["sample_every_n_seconds"]):
            frame   = resize_frame(frame, max_dim)   # ← resize HERE before anything
            frame_count += 1
            if frame_count % 50 == 0:
                pct = int((frame_idx / max(total, 1)) * 100)
                print(f"    frame {frame_idx}  ({pct}% of video)", end="\r")
            img_h, img_w = frame.shape[:2]
            results = model(frame, verbose=False, device=device)[0]

            centroids = []
            frame_dets = []

            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = CLASSES[cls_id]
                conf     = float(box.conf[0])

                if cls_name not in VEHICLE_CLS:
                    continue   # ignore helmets/plates in tracking pass

                if conf < CONFIG["conf_discard"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1

                if bw < CONFIG["min_vehicle_width"] or bh < CONFIG["min_vehicle_height"]:
                    continue

                cx_px = (x1 + x2) // 2
                cy_px = (y1 + y2) // 2
                centroids.append((cx_px, cy_px))

            tracker.update(centroids, frame_idx)
            frame_store[frame_idx] = frame  # stored already resized

        # ── Smart frame selection per track ──
        selected_frames = set()
        for track_id, history in tracker.track_history.items():
            chosen = select_track_frames(history)
            selected_frames.update(chosen)

        print(f"\n  Smart sampling: {len(selected_frames)} frames selected from {frame_count} streamed")

        # ── TWO-STAGE DETECTION + CLASSIFY ────────────────────
        file_counter  = 0
        accepted_items = []

        for frame_idx in sorted(selected_frames):
            frame    = frame_store[frame_idx]
            img_h, img_w = frame.shape[:2]

            # ── STAGE 1: detect vehicles on full frame ────────
            s1_results = model(frame, verbose=False, device=CONFIG["device"])[0]
            vehicle_dets = []   # (cls_id, x1, y1, x2, y2, conf)  pixel coords

            for box in s1_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES[cls_id]
                if cls_name not in VEHICLE_CLS:
                    continue   # Stage 1 only cares about vehicles

                conf = float(box.conf[0])
                if conf < CONFIG["conf_discard"]:
                    stats["discarded"] += 1
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = x2 - x1, y2 - y1

                if bw < CONFIG["min_vehicle_width"] or bh < CONFIG["min_vehicle_height"]:
                    stats["discarded"] += 1
                    continue   # vehicle too small / far away

                vehicle_dets.append((cls_id, x1, y1, x2, y2, conf))

            if not vehicle_dets:
                continue   # no vehicles in this frame, skip entirely

            # Skip blurry frames
            if is_blurry(frame, threshold=60):
                stats["discarded"] += len(vehicle_dets)
                continue

            # ── STAGE 2: for each vehicle crop, detect small objects ──
            # Final detections for this frame: vehicles + their sub-objects
            # stored in YOLO normalised format (cls_id, cx, cy, w, h, conf)
            all_dets = []
            needs_review = False

            for v_cls_id, vx1, vy1, vx2, vy2, v_conf in vehicle_dets:
                # Add the vehicle itself
                vcx, vcy, vw, vh = box_to_yolo(vx1, vy1, vx2, vy2, img_w, img_h)
                all_dets.append((v_cls_id, vcx, vcy, vw, vh, v_conf))
                if v_conf < CONFIG["conf_auto_accept"]:
                    needs_review = True

                # Crop the vehicle ROI (with padding)
                pad  = CONFIG["crop_padding"]
                rx1  = max(0,      vx1 - pad)
                ry1  = max(0,      vy1 - pad)
                rx2  = min(img_w,  vx2 + pad)
                ry2  = min(img_h,  vy2 + pad)
                crop = frame[ry1:ry2, rx1:rx2]

                if crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]

                # Run YOLO on crop
                s2_results = model(crop, verbose=False, device=device)[0]

                for cbox in s2_results.boxes:
                    c_cls_id  = int(cbox.cls[0])
                    c_cls_name = CLASSES[c_cls_id]

                    if c_cls_name not in SMALL_CLS:
                        continue   # Stage 2 only keeps small objects

                    c_conf = float(cbox.conf[0])
                    if c_conf < CONFIG["conf_stage2_min"]:
                        continue

                    # Pixel coords inside crop
                    cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])

                    # Map back to full-frame pixel coords
                    fx1 = rx1 + cx1
                    fy1 = ry1 + cy1
                    fx2 = rx1 + cx2
                    fy2 = ry1 + cy2

                    # Normalise to full frame
                    ncx, ncy, nw, nh = box_to_yolo(fx1, fy1, fx2, fy2, img_w, img_h)
                    all_dets.append((c_cls_id, ncx, ncy, nw, nh, c_conf))

                    if c_conf < CONFIG["conf_auto_accept"]:
                        needs_review = True

            if not all_dets:
                continue

            file_id = f"{vid_id}_{file_counter:06d}"
            file_counter += 1

            if needs_review:
                review_q.append({
                    "frame": frame, "detections": all_dets,
                    "img_h": img_h, "img_w": img_w,
                    "auto": False, "file_id": file_id
                })
            else:
                accepted_items.append({
                    "frame": frame, "detections": all_dets,
                    "img_h": img_h, "img_w": img_w,
                    "auto": True, "file_id": file_id
                })
                stats["auto_accepted"] += 1

        # ── Human review ──
        if review_q:
            print(f"\n  → {len(review_q)} frame(s) need your review.")
            print("    Opening review window...")
            print("    Shortcuts: [A] Accept  [S] Skip  [E] Edit  [N] Next  [Q] Quit\n")
            review_results = ui.review_batch(review_q)

            for item, result in zip(review_q, review_results):
                if result["accepted"]:
                    item["detections"] = result["detections"]
                    accepted_items.append(item)
                    stats["reviewed"] += 1
                else:
                    stats["discarded"] += 1

        # ── Save accepted frames ──
        print(f"\n  Saving {len(accepted_items)} accepted frame(s)...")
        for item in accepted_items:
            frame    = item["frame"]
            dets     = item["detections"]   # (cls_id, cx, cy, w, h, conf) normalised
            file_id  = item["file_id"]
            img_h, img_w = item["img_h"], item["img_w"]

            split    = "val" if random.random() < CONFIG["val_split"] else "train"
            img_name = f"{file_id}.jpg"
            lbl_name = f"{file_id}.txt"

            img_out  = os.path.join(folders[f"full_images_{split}"], img_name)
            lbl_out  = os.path.join(folders[f"full_labels_{split}"], lbl_name)

            cv2.imwrite(img_out, frame)
            # Save all detections (vehicles + sub-objects) in full-frame coords
            save_label(lbl_out, [(d[0], d[1], d[2], d[3], d[4]) for d in dets])

            # ── Crop each vehicle + re-normalize sub-object labels to crop ──
            for v_det in dets:
                v_cls_id, vcx, vcy, vw, vh, v_conf = v_det
                if CLASSES[v_cls_id] not in VEHICLE_CLS:
                    continue

                # Vehicle pixel box in full frame
                vx1, vy1, vx2, vy2 = yolo_to_box(vcx, vcy, vw, vh, img_w, img_h)
                pad  = CONFIG["crop_padding"]
                rx1  = max(0,     vx1 - pad)
                ry1  = max(0,     vy1 - pad)
                rx2  = min(img_w, vx2 + pad)
                ry2  = min(img_h, vy2 + pad)
                crop = frame[ry1:ry2, rx1:rx2]

                if crop.size == 0:
                    continue

                crop_h, crop_w = crop.shape[:2]
                v_cls_name = CLASSES[v_cls_id]
                crop_img_name = f"{file_id}_{v_cls_name}.jpg"
                crop_lbl_name = f"{file_id}_{v_cls_name}.txt"

                c_img_out = os.path.join(folders[f"crop_images_{split}"], crop_img_name)
                c_lbl_out = os.path.join(folders[f"crop_labels_{split}"], crop_lbl_name)

                cv2.imwrite(c_img_out, crop)

                # Re-normalize sub-object coords from full frame → crop space
                crop_dets = []

                # Include the vehicle itself (re-normalised to crop)
                nvx1 = max(0, vx1 - rx1)
                nvy1 = max(0, vy1 - ry1)
                nvx2 = min(crop_w, vx2 - rx1)
                nvy2 = min(crop_h, vy2 - ry1)
                ncx, ncy, nw2, nh2 = box_to_yolo(nvx1, nvy1, nvx2, nvy2, crop_w, crop_h)
                crop_dets.append((v_cls_id, ncx, ncy, nw2, nh2))

                # Sub-objects: re-normalise from full frame to crop coords
                for sd in dets:
                    s_cls_id, scx, scy, sw, sh, _ = sd
                    if CLASSES[s_cls_id] not in SMALL_CLS:
                        continue
                    # Full frame pixel coords
                    sx1, sy1, sx2, sy2 = yolo_to_box(scx, scy, sw, sh, img_w, img_h)
                    # Clip to crop region
                    nx1 = max(rx1, sx1) - rx1
                    ny1 = max(ry1, sy1) - ry1
                    nx2 = min(rx2, sx2) - rx1
                    ny2 = min(ry2, sy2) - ry1
                    if nx2 > nx1 and ny2 > ny1:
                        ncx2, ncy2, nw3, nh3 = box_to_yolo(nx1, ny1, nx2, ny2, crop_w, crop_h)
                        crop_dets.append((s_cls_id, ncx2, ncy2, nw3, nh3))

                save_label(c_lbl_out, crop_dets)

        # ── Mark video as done ──
        session["processed_videos"].append(vid_path.name)
        save_session(folders["session"], session)
        print(f"  Done: {vid_path.name}")

    # ── Write dataset.yaml ──
    yaml_path = os.path.join(CONFIG["output_folder"], "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {CONFIG['output_folder']}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    # ── Final summary ──
    print("\n" + "="*60)
    print("  DONE!")
    print(f"  Auto-accepted : {stats['auto_accepted']}")
    print(f"  Human-reviewed: {stats['reviewed']}")
    print(f"  Discarded     : {stats['discarded']}")
    print(f"\n  Dataset saved to: {CONFIG['output_folder']}")
    print(f"  YAML file      : {yaml_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_pipeline()
