import os
import shutil

# Source folder
raw_dir = "D:\\gsfc\\project\\sem8_project\\Project_yolo model training\\raw"

# Destination folders
images_dir = "D:\\gsfc\\project\\sem8_project\\Project_yolo model training\\images_all"
labels_dir = "D:\\gsfc\\project\\sem8_project\\Project_yolo model training\\labels_all"

# Create destination folders if not exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Loop through all files in raw/
for file in os.listdir(raw_dir):
    src_path = os.path.join(raw_dir, file)

    if os.path.isfile(src_path):
        file_lower = file.lower()

        # Move images
        if any(file_lower.endswith(ext) for ext in image_exts):
            shutil.copy(src_path, os.path.join(images_dir, file))

        # Move labels
        elif file_lower.endswith(".txt"):
            shutil.copy(src_path, os.path.join(labels_dir, file))

print("Done: Images and labels separated.")