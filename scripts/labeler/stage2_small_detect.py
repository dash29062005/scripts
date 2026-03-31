"""
=============================================================
  STAGE 2 — Small Object Detection (Helmet / Plate)
=============================================================
  INPUT:
    /crops/images/  → sharp vehicle crops from Stage 1
    /crops/labels/  → car/bike label (crop coords)
    /frames/labels/ → car/bike labels (frame coords) — will be UPDATED

  OUTPUT:
    /crops/labels/  → UPDATED: adds helmet/plate in crop coords
    /frames/labels/ → UPDATED: adds helmet/plate mapped back to frame
    /edit_queue/    → uncertain frames sent for manual LabelImg review

  FLOW:
    For each crop in /crops/images/:
      → YOLO on sharp crop → detect helmet / without_helmet / number_plate
      → conf >= 0.60  → auto accept
      → 0.25–0.60     → show review window (crop image)
      → press E       → save crop + frame to /edit_queue/
      → map accepted coords from crop space → frame space
      → UPDATE /frames/labels/ with new small object boxes
      → UPDATE /crops/labels/  with small object boxes in crop space

  REVIEW KEYS (shown on crop image):
    A / N  → Accept detections shown
    S      → Skip this crop (discard uncertain detections)
    E      → Send to edit_queue for manual LabelImg labeling
    Q      → Quit and save progress

  RUN:  python stage2_small_detect.py
        (run AFTER stage1_extract.py)
=============================================================
"""

import cv2
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
    "output_folder": r"D:\gsfc\project\sem8_project\Project_yolo model training\raw",

    "classes":       ["car", "bike", "helmet", "without_helmet", "number_plate"],

    # Auto-accept small object detections above this
    "conf_auto_accept": 0.60,

    # Minimum confidence to even consider (below = discard)
    "conf_stage2_min":  0.25,

    # GPU: 0 = GTX 1650   |   "cpu" = CPU
    "device":        0,

    # JPEG quality for edit_queue copies
    "jpg_quality":   95,
}
# ══════════════════════════════════════════════════════════

CLASSES     = CONFIG["classes"]
VEHICLE_CLS = {"car", "bike"}
SMALL_CLS   = {"helmet", "without_helmet", "number_plate"}

CLS_COLOR = {
    "car":            (0,  200,  80),
    "bike":           (0,  180, 120),
    "helmet":         (0,  140, 255),
    "without_helmet": (0,   60, 220),
    "number_plate":   (200, 180,   0),
}


# ──────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────

def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1)       / img_w
    h  = (y2 - y1)       / img_h
    return round(cx,6), round(cy,6), round(w,6), round(h,6)


def yolo_to_box(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return x1, y1, x2, y2


def read_label(path):
    """Returns list of (cls_id, cx, cy, w, h)"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append((int(parts[0]),
                              float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
    return rows


def save_label(path, dets):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dets:
            f.write(f"{d[0]} {d[1]:.6f} {d[2]:.6f} {d[3]:.6f} {d[4]:.6f}\n")


def save_img(path, img, quality=95):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def setup_folders(base):
    paths = {
        "crops_img":   Path(base) / "crops"      / "images",
        "crops_lbl":   Path(base) / "crops"      / "labels",
        "frames_img":  Path(base) / "frames"     / "images",
        "frames_lbl":  Path(base) / "frames"     / "labels",
        "edit_queue":  Path(base) / "edit_queue",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in paths.items()}


def load_session(base):
    p = Path(base) / "stage2_session.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done_crops": [], "stats": {"auto": 0, "reviewed": 0, "edit_queued": 0, "discarded": 0}}


def save_session(base, sess):
    p = Path(base) / "stage2_session.json"
    p.write_text(json.dumps(sess, indent=2))


# ──────────────────────────────────────────────────────────
#  CROP NAME → FRAME NAME
# ──────────────────────────────────────────────────────────

def crop_name_to_frame_name(crop_stem):
    """
    Crop filename format: {vid_id}_{frame_idx:07d}_c{n}_{classname}
    Frame filename format: {vid_id}_{frame_idx:07d}
    Extract frame base name from crop name.
    """
    parts = crop_stem.split("_c")
    if len(parts) >= 2:
        return parts[0]
    return None


def get_crop_origin(crop_stem, frames_lbl_dir, crops_lbl_dir, crop_img):
    """
    Recover rx1,ry1 (crop origin in frame) from the vehicle label in frame
    and the vehicle label in crop.
    Returns (rx1, ry1, frame_w, frame_h) or None if not recoverable.
    """
    frame_stem = crop_name_to_frame_name(crop_stem)
    if not frame_stem:
        return None

    frame_lbl_path = os.path.join(frames_lbl_dir, frame_stem + ".txt")
    crop_lbl_path  = os.path.join(crops_lbl_dir,  crop_stem  + ".txt")

    frame_dets = read_label(frame_lbl_path)
    crop_dets  = read_label(crop_lbl_path)

    if not frame_dets or not crop_dets:
        return None

    # Match vehicle box between frame and crop to recover origin
    # The crop label has exactly 1 vehicle box (the vehicle this crop came from)
    crop_h, crop_w = crop_img.shape[:2]
    c_cls, ccx, ccy, cw, ch = crop_dets[0]

    # Vehicle pixel box in crop space
    cx1, cy1, cx2, cy2 = yolo_to_box(ccx, ccy, cw, ch, crop_w, crop_h)
    # Center in crop space
    c_center_x = (cx1 + cx2) / 2
    c_center_y = (cy1 + cy2) / 2

    # Find matching vehicle in frame labels (same class, closest center)
    # We need frame image size — infer from label coords not available directly.
    # Use the frame image file if it exists.
    # Store frame_img_dir alongside to get frame size.
    return (c_cls, ccx, ccy, cw, ch, crop_w, crop_h, frame_stem)


# ──────────────────────────────────────────────────────────
#  REVIEW UI
# ──────────────────────────────────────────────────────────

class ReviewUI:
    def __init__(self):
        self.drawing = False
        self.ix = self.iy = -1
        self.rect = None
        self.edit_mode = False
        self.current_display = None

    def mouse_cb(self, event, x, y, flags, param):
        if not self.edit_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            tmp = self.current_display.copy()
            cv2.rectangle(tmp, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("STAGE2 REVIEW", tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect = (min(self.ix,x), min(self.iy,y),
                         max(self.ix,x), max(self.iy,y))

    def _render(self, img, dets, img_w, img_h, hud):
        disp = img.copy()
        for cls_id, cx, cy, w, h, conf in dets:
            x1,y1,x2,y2 = yolo_to_box(cx,cy,w,h,img_w,img_h)
            color = CLS_COLOR.get(CLASSES[cls_id], (200,200,200))
            cv2.rectangle(disp,(x1,y1),(x2,y2),color,2)
            cv2.putText(disp, f"{CLASSES[cls_id]} {conf:.2f}",
                        (x1, max(y1-6,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.rectangle(disp,(0,img_h-28),(img_w,img_h),(30,30,30),-1)
        cv2.putText(disp, hud, (8,img_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220,220,220), 1)
        return disp

    def _pick_class(self, base_img):
        overlay = base_img.copy()
        img_h, img_w = overlay.shape[:2]
        px, py    = 30, 30
        panel_w   = 320
        panel_h   = 50 + len(CLASSES) * 36 + 16
        cv2.rectangle(overlay,(px,py),(px+panel_w,py+panel_h),(20,20,20),-1)
        cv2.rectangle(overlay,(px,py),(px+panel_w,py+panel_h),(160,160,160),1)
        cv2.putText(overlay,"Choose class:",(px+12,py+28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),1)
        for idx, cls_name in enumerate(CLASSES):
            yp    = py + 52 + idx*36
            color = CLS_COLOR.get(cls_name,(200,200,200))
            cv2.rectangle(overlay,(px+10,yp-18),(px+44,yp+6),color,-1)
            cv2.putText(overlay,str(idx),(px+20,yp),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(10,10,10),2)
            cv2.putText(overlay,f"  {cls_name}",(px+50,yp),
                        cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),1)
        cv2.putText(overlay,"ESC = cancel",(px+12,py+panel_h-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,140,140),1)
        cv2.imshow("STAGE2 REVIEW", overlay)
        while True:
            k = cv2.waitKey(20) & 0xFF
            if ord("0") <= k <= ord("0")+len(CLASSES)-1:
                return k - ord("0")
            if k == 27:
                return -1

    def review(self, crop_img, dets):
        """
        Show crop image with detections.
        Returns: ("accept", dets) | ("skip", []) | ("edit", []) | ("quit", [])
        """
        img_h, img_w = crop_img.shape[:2]
        hud = "A/N=Accept  S=Skip  E=Send to edit_queue  Q=Quit"

        cv2.namedWindow("STAGE2 REVIEW", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("STAGE2 REVIEW", min(img_w*2, 1280), min(img_h*2, 720))
        cv2.setMouseCallback("STAGE2 REVIEW", self.mouse_cb)

        self.edit_mode = False
        self.rect      = None
        display = self._render(crop_img, dets, img_w, img_h, hud)
        self.current_display = display
        cv2.imshow("STAGE2 REVIEW", display)

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("a"), ord("n")):
                return "accept", dets

            elif key == ord("s"):
                return "skip", []

            elif key == ord("e"):
                return "edit", []

            elif key == ord("q"):
                cv2.destroyAllWindows()
                return "quit", []


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STAGE 2 — Small Object Detection (Helmet / Plate)")
    print("="*60)

    base = CONFIG["output_folder"]

    if not os.path.exists(CONFIG["model_path"]):
        print(f"\n[ERROR] Model not found: {CONFIG['model_path']}\n")
        sys.exit(1)

    folders = setup_folders(base)
    session = load_session(base)
    stats   = session["stats"]

    print(f"\n  Loading model on device={CONFIG['device']} ...")
    model  = YOLO(CONFIG["model_path"])
    ui     = ReviewUI()
    device = CONFIG["device"]

    # All crop images
    crop_imgs = sorted(Path(folders["crops_img"]).glob("*.jpg"))
    done_set  = set(session["done_crops"])
    pending   = [p for p in crop_imgs if p.stem not in done_set]

    print(f"  Crops to process: {len(pending)}")
    print(f"  Keyboard: A=Accept  S=Skip  E=EditQueue  Q=Quit\n")

    quit_flag = False

    for crop_path in pending:
        if quit_flag:
            break

        crop_stem  = crop_path.stem
        frame_stem = crop_name_to_frame_name(crop_stem)

        if not frame_stem:
            print(f"  [WARN] Cannot parse frame name from: {crop_stem}")
            session["done_crops"].append(crop_stem)
            continue

        # Load crop image
        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            session["done_crops"].append(crop_stem)
            continue

        crop_h, crop_w = crop_img.shape[:2]

        # Load crop's existing vehicle label to get crop origin in frame
        crop_lbl_path  = os.path.join(folders["crops_lbl"],  crop_stem  + ".txt")
        frame_lbl_path = os.path.join(folders["frames_lbl"], frame_stem + ".txt")
        frame_img_path = os.path.join(folders["frames_img"], frame_stem + ".jpg")

        crop_dets  = read_label(crop_lbl_path)   # vehicle in crop coords
        frame_dets = read_label(frame_lbl_path)  # vehicles in frame coords

        # Need frame size to map coords back
        frame_img = cv2.imread(frame_img_path)
        if frame_img is None:
            session["done_crops"].append(crop_stem)
            continue
        frame_h, frame_w = frame_img.shape[:2]

        # Recover crop origin (rx1, ry1) in frame pixels
        # The crop label has 1 vehicle. The frame label has the same vehicle.
        # vehicle center in crop → vehicle center in frame → origin = center_frame - center_crop
        if not crop_dets or not frame_dets:
            session["done_crops"].append(crop_stem)
            continue

        # Get vehicle class from crop label (first entry)
        v_cls = crop_dets[0][0]
        # Vehicle center in crop (pixels)
        vcx_c, vcy_c, vw_c, vh_c = crop_dets[0][1], crop_dets[0][2], crop_dets[0][3], crop_dets[0][4]
        vc_px = vcx_c * crop_w
        vc_py = vcy_c * crop_h

        # Find matching vehicle in frame labels (same class)
        best_match = None
        for fd in frame_dets:
            if fd[0] == v_cls:
                best_match = fd
                break
        if best_match is None:
            session["done_crops"].append(crop_stem)
            continue

        # Vehicle center in frame (pixels)
        vfcx, vfcy = best_match[1] * frame_w, best_match[2] * frame_h
        # Crop origin in frame
        rx1 = int(vfcx - vc_px)
        ry1 = int(vfcy - vc_py)

        # ── Run YOLO on sharp crop ─────────────────────────
        results = model(crop_img, verbose=False, device=device)[0]

        small_dets = []   # (cls_id, cx, cy, w, h, conf) in crop coords

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = CLASSES[cls_id]

            if cls_name not in SMALL_CLS:
                continue

            conf = float(box.conf[0])
            if conf < CONFIG["conf_stage2_min"]:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx,cy,w,h    = box_to_yolo(x1,y1,x2,y2,crop_w,crop_h)
            small_dets.append((cls_id, cx, cy, w, h, conf))

        if not small_dets:
            # No small objects found — mark done, move on
            session["done_crops"].append(crop_stem)
            save_session(base, session)
            del frame_img
            continue

        # ── Classify: auto-accept vs review ───────────────
        auto_dets   = [d for d in small_dets if d[5] >= CONFIG["conf_auto_accept"]]
        review_dets = [d for d in small_dets if d[5] <  CONFIG["conf_auto_accept"]]

        accepted_crop_dets = list(auto_dets)   # confirmed detections in crop coords

        if review_dets:
            print(f"\n  Review: {crop_stem}")
            action, result_dets = ui.review(crop_img, review_dets)

            if action == "quit":
                quit_flag = True
                del frame_img
                break

            elif action == "accept":
                accepted_crop_dets.extend(result_dets)
                stats["reviewed"] += 1

            elif action == "edit":
                # Save crop and original frame to edit_queue for LabelImg
                eq_crop_img  = os.path.join(folders["edit_queue"], crop_stem + ".jpg")
                eq_crop_lbl  = os.path.join(folders["edit_queue"], crop_stem + ".txt")
                eq_frame_img = os.path.join(folders["edit_queue"], frame_stem + "_FRAME.jpg")
                eq_frame_lbl = os.path.join(folders["edit_queue"], frame_stem + "_FRAME.txt")

                cv2.imwrite(eq_crop_img,  crop_img,  [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpg_quality"]])
                cv2.imwrite(eq_frame_img, frame_img, [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpg_quality"]])

                # Write existing labels as starting point for LabelImg
                save_label(eq_crop_lbl,  [(d[0],d[1],d[2],d[3],d[4]) for d in crop_dets])
                save_label(eq_frame_lbl, frame_dets)

                # Also copy classes.txt to edit_queue
                cls_src = Path(base) / "classes.txt"
                cls_dst = Path(folders["edit_queue"]) / "classes.txt"
                if cls_src.exists() and not cls_dst.exists():
                    cls_dst.write_text(cls_src.read_text())

                stats["edit_queued"] += 1
                print(f"  → Sent to edit_queue: {crop_stem}")
                session["done_crops"].append(crop_stem)
                save_session(base, session)
                del frame_img
                continue

            elif action == "skip":
                stats["discarded"] += len(review_dets)

        if auto_dets:
            stats["auto"] += len(auto_dets)

        # ── Map accepted crop detections → frame space ─────
        new_frame_small = []
        for cls_id, cx, cy, w, h, conf in accepted_crop_dets:
            # Pixel box in crop
            x1c = (cx - w/2) * crop_w
            y1c = (cy - h/2) * crop_h
            x2c = (cx + w/2) * crop_w
            y2c = (cy + h/2) * crop_h
            # Map to frame pixels
            x1f = int(rx1 + x1c)
            y1f = int(ry1 + y1c)
            x2f = int(rx1 + x2c)
            y2f = int(ry1 + y2c)
            # Clamp to frame bounds
            x1f = max(0, min(frame_w, x1f))
            y1f = max(0, min(frame_h, y1f))
            x2f = max(0, min(frame_w, x2f))
            y2f = max(0, min(frame_h, y2f))
            if x2f > x1f and y2f > y1f:
                fcx,fcy,fw,fh = box_to_yolo(x1f,y1f,x2f,y2f,frame_w,frame_h)
                new_frame_small.append((cls_id, fcx, fcy, fw, fh))

        # ── Update frame label (append small objects) ──────
        if new_frame_small:
            merged_frame = frame_dets + new_frame_small
            save_label(frame_lbl_path, merged_frame)

        # ── Update crop label (append small objects) ───────
        if accepted_crop_dets:
            merged_crop = crop_dets + [(d[0],d[1],d[2],d[3],d[4]) for d in accepted_crop_dets]
            save_label(crop_lbl_path, merged_crop)

        session["done_crops"].append(crop_stem)
        save_session(base, session)
        del frame_img

    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("  STAGE 2 COMPLETE")
    print(f"  Auto-accepted : {stats['auto']}")
    print(f"  Human-reviewed: {stats['reviewed']}")
    print(f"  Edit-queued   : {stats['edit_queued']}")
    print(f"  Discarded     : {stats['discarded']}")
    if stats["edit_queued"] > 0:
        print(f"\n  → Run manual_label.py to label edit_queue in LabelImg")
        print(f"  → Then run stage3_merge_edits.py to merge back")
    else:
        print(f"\n  → Run stage3_merge_edits.py to finalize dataset")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
