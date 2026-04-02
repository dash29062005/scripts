"""
Script 2 — select.py
====================
Reads the candidate frames saved by extractor.py and applies a
quality-selection pass WITHOUT re-running YOLO:

  1. Score each detection by how centered + how large it is.
  2. IoU dedup — skip a frame if its best detection overlaps
     heavily with an already-accepted detection (removes near-duplicates).
  3. Copy the top-scoring frames to the final dataset folder.

Run AFTER extractor.py has finished.

Tunable parameters are all in CONFIG below.
"""

import os
import shutil
import json
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────── CONFIG ────────────────────────────────────────
CONFIG = {
    # Where Script 1 saved candidates
    "candidate_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\candidates",

    # Where Script 2 writes the curated dataset
    "final_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\only_frames",

    # ── Deduplication ────────────────────────────────────────
    # If the IoU between a candidate box and ANY already-accepted box
    # exceeds this threshold, the frame is skipped.
    # Lower = keep more frames (less aggressive dedup)
    # Higher = keep fewer frames (more aggressive dedup)
    "iou_threshold": 0.50,

    # ── Center-bias weight ───────────────────────────────────
    # Score = area_score * (1 - center_weight) + center_score * center_weight
    # 0.0 = pure area scoring  |  1.0 = pure center scoring
    "center_weight": 0.40,

    # ── Edge margin filter ───────────────────────────────────
    # Reject detections whose center is within this fraction of
    # the frame edge (catches entry/exit frames)
    # e.g. 0.10 = ignore anything in the outer 10% strip
    "edge_margin": 0.10,

    # ── Max frames per video ─────────────────────────────────
    # Caps dataset contribution per video clip.
    # Set to None to keep all accepted frames.
    "max_frames_per_video": 50,

    "classes": ["car", "bike", "helmet", "without_helmet", "number_plate"],
}

# ───────────────────────── DATA TYPES ──────────────────────────────────────

class Detection(NamedTuple):
    cls_id: int
    cx: float   # YOLO normalised [0,1]
    cy: float
    bw: float
    bh: float


class ScoredFrame(NamedTuple):
    score: float
    name: str           # filename stem (no extension)
    best_det: Detection # highest-scoring detection in this frame


# ───────────────────────── HELPERS ─────────────────────────────────────────

def load_label(label_path: str) -> list[Detection]:
    """Parse a YOLO .txt label file into a list of Detection objects."""
    dets = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            dets.append(Detection(cls_id, cx, cy, bw, bh))
    return dets


def score_detection(det: Detection, edge_margin: float, center_weight: float) -> float:
    """
    Return a quality score in [0, 1] for a single detection.

    Two components:
      area_score   — bigger box = better (more pixels = more detail)
      center_score — how close is the box center to the frame center?

    Edge-margin check: if the box center is too close to any frame edge,
    return -1.0 so the caller can discard this detection outright.
    """
    cx, cy, bw, bh = det.cx, det.cy, det.bw, det.bh

    # Edge-margin reject
    if cx < edge_margin or cx > 1 - edge_margin:
        return -1.0
    if cy < edge_margin or cy > 1 - edge_margin:
        return -1.0

    # Area score: normalised box area, capped at 0.5 of frame
    area_score = min(bw * bh / 0.5, 1.0)

    # Center score: Euclidean distance from frame center, inverted
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    max_dist = (0.5 ** 2 + 0.5 ** 2) ** 0.5      # corner distance ≈ 0.707
    center_score = 1.0 - (dist / max_dist)

    return area_score * (1 - center_weight) + center_score * center_weight


def iou(a: Detection, b: Detection) -> float:
    """
    Compute IoU between two YOLO-format detections.
    Both are in normalised [0,1] coordinates so the result is scale-invariant.
    """
    # Convert cx,cy,bw,bh → x1,y1,x2,y2
    def to_corners(d):
        return d.cx - d.bw / 2, d.cy - d.bh / 2, d.cx + d.bw / 2, d.cy + d.bh / 2

    ax1, ay1, ax2, ay2 = to_corners(a)
    bx1, by1, bx2, by2 = to_corners(b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlaps_any_accepted(det: Detection, accepted: list[Detection], threshold: float) -> bool:
    """Return True if det overlaps with any already-accepted detection above threshold."""
    for acc in accepted:
        if iou(det, acc) >= threshold:
            return True
    return False


# ───────────────────────── MAIN ────────────────────────────────────────────

def main():
    cand_img = Path(CONFIG["candidate_folder"]) / "images"
    cand_lbl = Path(CONFIG["candidate_folder"]) / "labels"

    if not cand_img.exists():
        print("❌ Candidate images folder not found. Run extractor.py first.")
        return

    final_img = Path(CONFIG["final_folder"]) / "images"
    final_lbl = Path(CONFIG["final_folder"]) / "labels"
    final_img.mkdir(parents=True, exist_ok=True)
    final_lbl.mkdir(parents=True, exist_ok=True)

    # Group files by video stem so we can apply per-video frame cap
    video_groups: dict[str, list[str]] = {}
    for img_file in sorted(cand_img.glob("*.jpg")):
        # Stem format: {vid_id}_{frame_idx:06d}
        # Split on the last underscore+6-digits suffix
        parts = img_file.stem.rsplit("_", 1)
        vid_id = parts[0] if len(parts) == 2 else img_file.stem
        video_groups.setdefault(vid_id, []).append(img_file.stem)

    total_accepted = 0
    total_rejected_score  = 0
    total_rejected_dedup  = 0
    summary: dict = {}

    for vid_id, stems in video_groups.items():
        print(f"\nCurating: {vid_id} ({len(stems)} candidates)")

        # ── Score every candidate frame ──────────────────────
        scored: list[ScoredFrame] = []
        for stem in stems:
            lbl_path = cand_lbl / (stem + ".txt")
            if not lbl_path.exists():
                continue

            dets = load_label(str(lbl_path))
            if not dets:
                continue

            # Pick the highest-scoring detection in this frame
            best_score = -1.0
            best_det   = None
            for det in dets:
                s = score_detection(
                    det,
                    CONFIG["edge_margin"],
                    CONFIG["center_weight"],
                )
                if s > best_score:
                    best_score = s
                    best_det   = det

            if best_score < 0 or best_det is None:
                total_rejected_score += 1
                continue

            scored.append(ScoredFrame(best_score, stem, best_det))

        # Sort best → worst
        scored.sort(key=lambda sf: sf.score, reverse=True)

        # ── IoU dedup + copy ─────────────────────────────────
        accepted_dets: list[Detection] = []
        accepted_count = 0
        cap = CONFIG["max_frames_per_video"]

        for sf in scored:
            if cap is not None and accepted_count >= cap:
                break

            if overlaps_any_accepted(sf.best_det, accepted_dets, CONFIG["iou_threshold"]):
                total_rejected_dedup += 1
                continue

            # Accept this frame
            accepted_dets.append(sf.best_det)
            accepted_count += 1

            src_img = cand_img / (sf.name + ".jpg")
            src_lbl = cand_lbl / (sf.name + ".txt")
            dst_img = final_img / (sf.name + ".jpg")
            dst_lbl = final_lbl / (sf.name + ".txt")

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

        total_accepted += accepted_count
        summary[vid_id] = {"candidates": len(stems), "accepted": accepted_count}
        print(f"  Accepted {accepted_count} / {len(stems)}")

    # ── Save summary ─────────────────────────────────────────
    report = {
        "total_accepted":        total_accepted,
        "total_rejected_score":  total_rejected_score,
        "total_rejected_dedup":  total_rejected_dedup,
        "per_video":             summary,
        "config_used":           CONFIG,
    }
    report_path = Path(CONFIG["final_folder"]) / "selection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Selection complete.")
    print(f"   Total accepted      : {total_accepted}")
    print(f"   Rejected (edge/size): {total_rejected_score}")
    print(f"   Rejected (dedup)    : {total_rejected_dedup}")
    print(f"   Report saved        → {report_path}")


if __name__ == "__main__":
    main()
