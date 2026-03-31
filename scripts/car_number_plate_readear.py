import cv2
import numpy as np
import pytesseract
import os
import glob

# ---------------- CONFIG ----------------
TEST_FOLDER = r"D:\gsfc\project\sem8_project\Project_yolo model training\test"

# ============ PREPROCESSING PARAMETERS ============
PREPROCESSING_CONFIG = {
    "target_width": 800,
    "gaussian_blur": (5, 5),       # Blur kernel for noise reduction
    "otsu_threshold": True,        # Use Otsu instead of adaptive
    "morph_kernel": (5, 5),        # Larger kernel for character connectivity
    "morph_close": 2,              # Closing iterations
    "morph_open": 1,               # Opening iterations
}

OCR_CONFIG = {
    "oem": 3,  # Legacy + LSTM combined
    "psm": 11,  # Sparse text - better for license plates with variable layout
    "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
}

VIZ_CONFIG = {
    "grid_spacing": 10,
    "window_width": 2000,
    "window_height": 1400,
    "label_font_size": 1.0,
}

# Try to find Tesseract installation
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Tesseract-OCR\tesseract.exe",
]

tesseract_found = False
for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Found Tesseract at: {path}\n")
        tesseract_found = True
        break

if not tesseract_found:
    print("ERROR: Tesseract OCR not found!")
    print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("Common installation paths:")
    for path in tesseract_paths:
        print(f"  - {path}")
    exit()

# Get all test images
image_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.jpg")) + glob.glob(os.path.join(TEST_FOLDER, "*.png")))
print(f"Found {len(image_files)} images in test folder\n")

if len(image_files) == 0:
    print("No images found in test folder!")
    exit()

# ====== PROCESS EACH IMAGE ======
for image_idx, image_path in enumerate(image_files, 1):
    print(f"\n{'='*80}")
    print(f"[{image_idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
    print(f"{'='*80}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    h_orig, w_orig = img.shape[:2]
    print(f"Original image size: {w_orig}x{h_orig}")

    # -------- Resize image for processing --------
    target_width = PREPROCESSING_CONFIG["target_width"]
    h, w = img.shape[:2]
    scale = target_width / w
    plate_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    print(f"Resized to: {plate_resized.shape[1]}x{plate_resized.shape[0]}")

    # -------- CV PREPROCESSING FOR OCR --------
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur for noise reduction (simpler than bilateral)
    blurred = cv2.GaussianBlur(gray, PREPROCESSING_CONFIG["gaussian_blur"], 0)

    # Step 3: Otsu's thresholding (works better for license plates)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological operations with larger kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, PREPROCESSING_CONFIG["morph_kernel"])

    # Opening to remove small noise
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=PREPROCESSING_CONFIG["morph_open"])

    # Closing to connect broken characters
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=PREPROCESSING_CONFIG["morph_close"])

    # Step 5: Dilation to enhance characters
    dilate = cv2.dilate(morph, kernel, iterations=1)

    # Step 6: Upscale for better OCR (Tesseract works better with larger characters)
    scale_factor = 2
    dilate_scaled = cv2.resize(dilate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # -------- OCR WITH IMPROVED CONFIG FOR 2-LINE PLATES --------
    # PSM 11: Sparse text (find text in no particular order) - better for plates
    # OEM 3: Legacy + LSTM combined mode
    ocr_config = (
        f"--oem {OCR_CONFIG['oem']} "
        f"--psm {OCR_CONFIG['psm']} "
        f"-c tessedit_char_whitelist={OCR_CONFIG['whitelist']}"
    )

    # Try OCR on upscaled dilated image (cleaner approach with simplified preprocessing)
    try:
        detected_text = pytesseract.image_to_string(dilate_scaled, config=ocr_config).strip()

        # Clean up: remove extra whitespace and newlines
        detected_text_clean = " / ".join(detected_text.split('\n'))

        print(f"\n  Detected: {detected_text_clean}")
    except Exception as e:
        print(f"\n  ERROR during OCR: {str(e)}")
        print("  Make sure Tesseract OCR is installed and path is correct.")
        detected_text_clean = "ERROR - OCR Failed"

    # -------- GRID DISPLAY WITH LABELS --------
    def to_bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if len(x.shape) == 2 else x

    def add_label(img, label_text, font_size=1):
        h, w = img.shape[:2]
        img_with_label = img.copy()
        cv2.putText(img_with_label, label_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
        return img_with_label

    # Create labeled images for grid
    img1 = add_label(plate_resized, "1. Original Image")
    img2 = add_label(to_bgr(gray), "2. Grayscale")
    img3 = add_label(to_bgr(blurred), "3. Gaussian Blur")
    img4 = add_label(to_bgr(thresh), "4. Otsu Threshold")
    img5 = add_label(to_bgr(morph), "5. Morphological Open+Close")
    img6 = add_label(to_bgr(dilate), "6. Dilated (For OCR)")

    # Create grid with spacing
    spacing = VIZ_CONFIG["grid_spacing"]
    h, w = img1.shape[:2]
    separator = np.ones((h, spacing, 3), dtype=np.uint8) * 255

    row1 = np.hstack([img1, separator, img2, separator, img3])
    row2 = np.hstack([img4, separator, img5, separator, img6])

    v_separator = np.ones((spacing, row1.shape[1], 3), dtype=np.uint8) * 255
    grid = np.vstack([row1, v_separator, row2])

    # Add title with detected text
    title_img = np.ones((80, grid.shape[1], 3), dtype=np.uint8) * 200
    cv2.putText(title_img, f"OCR Result: {detected_text_clean}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(title_img, f"Image: {os.path.basename(image_path)}", (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    final_grid = np.vstack([title_img, grid])

    # Create resizable window and display at large size
    window_name = f"[{image_idx}] Direct OCR - Processing Steps"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, VIZ_CONFIG["window_width"], VIZ_CONFIG["window_height"])
    cv2.imshow(window_name, final_grid)

    print("Press any key to continue to next image...")
    cv2.waitKey(0)

# Cleanup
print("\n" + "="*80)
print("Processing complete. Press any key to close all windows.")
print("="*80)
cv2.waitKey(0)
cv2.destroyAllWindows()
