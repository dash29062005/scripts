"""
=============================================================
  STAGE 1 — Vehicle Detection & Frame/Crop Extraction
=============================================================
  INPUT : videos in CONFIG["video_folder"]

  OUTPUT:
    /frames/images/  → original full-res frames (.jpg)
    /frames/labels/  → car/bike only label (.txt, YOLO format)
    /crops/images/   → sharp vehicle crop from original frame
    /crops/labels/   → car or bike label renormalized to crop coords
    classes.txt      → class index file for LabelImg

  FLOW:
    Video (chunked 200 frames at a time to protect RAM)
      → original frame → Stage 1 YOLO → detect car/bike
      → save original frame + vehicle label
      → crop vehicle from original frame (sharp, full-res)
      → renormalize vehicle box to crop coords
      → save crop + crop label (only car/bike class)

  NOTE: helmet/plate NOT detected here.
        Stage 2 will detect them on the sharp crops.

  RUN:  python stage1_extract.py
=============================================================
"""

import cv2
import gc
import json
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] Run:  pip install ultralytics\n")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
CONFIG = {
    "model_path":    r"D:\gsfc\project\sem8_project\Project_yolo model training\model\p1.pt",
    "video_folder":  r"D:\gsfc\project\sem8_project\Project_yolo model training\cctv",
    "output_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\raw",

    "classes":       ["car", "bike", "helmet", "without_helmet", "number_plate"],

    # Minimum confidence to keep a vehicle detection
    "conf_min":      0.30,

    # Skip vehicles smaller than this (in original frame pixels)
    "min_vehicle_w": 80,
    "min_vehicle_h": 80,

    # Padding (pixels) added around vehicle when cropping
    "crop_pad":      30,

    # Sample 1 frame every N seconds
    "sample_every_n_seconds": 1,

    # Max frames held in RAM at once (150 × ~25MB 4K ≈ 3.75GB peak)
    "chunk_size":    150,

    # Blur rejection threshold (higher = stricter)
    "blur_threshold": 80.0,

    # GPU: 0 = GTX 1650   |   "cpu" = CPU only
    "device":        0,

    # JPEG quality for saved images (95 = near-lossless, keeps plate detail)
    "jpg_quality":   95,
}
# ══════════════════════════════════════════════════════════

CLASSES     = CONFIG["classes"]
VEHICLE_CLS = {"car", "bike"}


# ──────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────

def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1)       / img_w
    h  = (y2 - y1)       / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def save_label(path, dets):
    """dets = [(cls_id, cx, cy, w, h), ...]"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dets:
            f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f}\n")


def save_img(path, img, quality=95):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def is_blurry(img, threshold):
    if img is None or img.size == 0:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def setup_folders(base):
    paths = {
        "frames_img": Path(base) / "frames" / "images",
        "frames_lbl": Path(base) / "frames" / "labels",
        "crops_img":  Path(base) / "crops"  / "images",
        "crops_lbl":  Path(base) / "crops"  / "labels",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    # classes.txt for LabelImg
    cls_file = Path(base) / "classes.txt"
    if not cls_file.exists():
        cls_file.write_text("\n".join(CLASSES) + "\n")
    return {k: str(v) for k, v in paths.items()}


def load_session(base):
    p = Path(base) / "stage1_session.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done_videos": [], "stats": {"frames": 0, "crops": 0, "discarded": 0}}


def save_session(base, sess):
    p = Path(base) / "stage1_session.json"
    p.write_text(json.dumps(sess, indent=2))


# ──────────────────────────────────────────────────────────
#  FRAME STREAM  (generator — 1 frame in memory at a time)
# ──────────────────────────────────────────────────────────

def stream_frames(video_path, every_n_seconds):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(fps * every_n_seconds))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx      = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            yield idx, frame, total
        idx += 1
    cap.release()


# ──────────────────────────────────────────────────────────
#  PROCESS ONE CHUNK
# ──────────────────────────────────────────────────────────

def process_chunk(chunk, model, folders, vid_id, stats):
    """
    chunk = [(frame_idx, original_frame), ...]
    - Runs YOLO on original frame (YOLO internally resizes to 640)
    - Saves original frame + vehicle label
    - Saves sharp crop + renormalized vehicle label
    """
    device  = CONFIG["device"]
    pad     = CONFIG["crop_pad"]
    quality = CONFIG["jpg_quality"]

    for frame_idx, orig_frame in chunk:
        img_h, img_w = orig_frame.shape[:2]

        # ── YOLO on original frame ─────────────────────────
        results = model(orig_frame, verbose=False, device=device)[0]

        vehicles = []   # (cls_id, x1, y1, x2, y2)

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = CLASSES[cls_id]

            if cls_name not in VEHICLE_CLS:
                continue

            conf = float(box.conf[0])
            if conf < CONFIG["conf_min"]:
                stats["discarded"] += 1
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < CONFIG["min_vehicle_w"] or \
               (y2 - y1) < CONFIG["min_vehicle_h"]:
                stats["discarded"] += 1
                continue

            vehicles.append((cls_id, x1, y1, x2, y2))

        if not vehicles:
            stats["discarded"] += 1
            continue

        # ── Blur check ─────────────────────────────────────
        if is_blurry(orig_frame, CONFIG["blur_threshold"]):
            stats["discarded"] += len(vehicles)
            continue

        # ── Save original frame ────────────────────────────
        base_name  = f"{vid_id}_{frame_idx:07d}"
        frame_img  = os.path.join(folders["frames_img"], base_name + ".jpg")
        frame_lbl  = os.path.join(folders["frames_lbl"], base_name + ".txt")

        save_img(frame_img, orig_frame, quality)

        # Frame label = vehicle boxes only (Stage 2 will ADD small objects later)
        frame_dets = []
        for cls_id, x1, y1, x2, y2 in vehicles:
            cx, cy, w, h = box_to_yolo(x1, y1, x2, y2, img_w, img_h)
            frame_dets.append((cls_id, cx, cy, w, h))
        save_label(frame_lbl, frame_dets)
        stats["frames"] += 1

        # ── Crop each vehicle from ORIGINAL frame ──────────
        for crop_n, (cls_id, x1, y1, x2, y2) in enumerate(vehicles):
            cls_name = CLASSES[cls_id]

            # Padded crop region (clamped to frame bounds)
            rx1 = max(0,      x1 - pad)
            ry1 = max(0,      y1 - pad)
            rx2 = min(img_w,  x2 + pad)
            ry2 = min(img_h,  y2 + pad)

            crop = orig_frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]

            # Renormalize vehicle box to crop coordinate space
            lx1 = max(0,      x1 - rx1)
            ly1 = max(0,      y1 - ry1)
            lx2 = min(crop_w, x2 - rx1)
            ly2 = min(crop_h, y2 - ry1)
            ccx, ccy, cw, ch = box_to_yolo(lx1, ly1, lx2, ly2, crop_w, crop_h)

            crop_name = f"{base_name}_c{crop_n}_{cls_name}"
            save_img(os.path.join(folders["crops_img"], crop_name + ".jpg"), crop, quality)
            save_label(os.path.join(folders["crops_lbl"], crop_name + ".txt"),
                       [(cls_id, ccx, ccy, cw, ch)])
            stats["crops"] += 1

    del chunk
    gc.collect()


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STAGE 1 — Vehicle Detection & Extraction")
    print("="*60)

    base = CONFIG["output_folder"]

    if not os.path.exists(CONFIG["model_path"]):
        print(f"\n[ERROR] Model not found: {CONFIG['model_path']}\n")
        sys.exit(1)
    if not os.path.isdir(CONFIG["video_folder"]):
        print(f"\n[ERROR] Video folder not found: {CONFIG['video_folder']}\n")
        sys.exit(1)

    folders = setup_folders(base)
    session = load_session(base)
    stats   = session["stats"]

    print(f"\n  Loading model on device={CONFIG['device']} ...")
    model = YOLO(CONFIG["model_path"])

    exts   = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV", ".AVI"}
    videos = sorted([
        f for f in Path(CONFIG["video_folder"]).iterdir()
        if f.suffix in exts and f.name not in session["done_videos"]
    ])

    if not videos:
        print("\n  No new videos to process. Done!\n")
        return

    print(f"  Videos to process: {len(videos)}")
    print(f"  Output: {base}\n")

    for vid_path in videos:
        vid_id = vid_path.stem[:12].replace(" ", "_")
        print(f"\n{'─'*55}")
        print(f"  {vid_path.name}  (id={vid_id})")
        print(f"{'─'*55}")

        chunk       = []
        chunk_num   = 0
        sampled     = 0

        for frame_idx, frame, total in stream_frames(vid_path, CONFIG["sample_every_n_seconds"]):
            chunk.append((frame_idx, frame))
            sampled += 1

            pct = int(frame_idx / max(total, 1) * 100)
            print(f"  Streaming... frame {frame_idx} ({pct}%)  sampled={sampled}", end="\r")

            if len(chunk) >= CONFIG["chunk_size"]:
                chunk_num += 1
                print(f"\n  → Processing chunk {chunk_num} ({len(chunk)} frames)...")
                process_chunk(chunk, model, folders, vid_id, stats)
                chunk = []

        if chunk:
            chunk_num += 1
            print(f"\n  → Processing final chunk ({len(chunk)} frames)...")
            process_chunk(chunk, model, folders, vid_id, stats)

        session["done_videos"].append(vid_path.name)
        save_session(base, session)

        print(f"\n  ✓ {vid_path.name}")
        print(f"    Frames saved : {stats['frames']}")
        print(f"    Crops saved  : {stats['crops']}")
        print(f"    Discarded    : {stats['discarded']}")

    print("\n" + "="*60)
    print("  STAGE 1 COMPLETE")
    print(f"  Frames : {stats['frames']}")
    print(f"  Crops  : {stats['crops']}")
    print(f"\n  Next → python stage2_small_detect.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
