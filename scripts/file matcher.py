import os

images_dir = "images_all"
labels_dir = "labels_all"

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Get image base names
image_files = [
    f for f in os.listdir(images_dir)
    if any(f.lower().endswith(ext) for ext in image_exts)
]
image_names = set(os.path.splitext(f)[0] for f in image_files)

# Get label base names
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
label_names = set(os.path.splitext(f)[0] for f in label_files)

# Matching pairs
valid_pairs = image_names & label_names

# Unmatched files
images_without_labels = image_names - label_names
labels_without_images = label_names - image_names

print(f"Valid pairs: {len(valid_pairs)}")
print(f"Images without labels: {len(images_without_labels)}")
print(f"Labels without images: {len(labels_without_images)}")

# Optional: print names
print("\nSample valid pairs:")
for name in list(valid_pairs)[:10]:
    print(name)