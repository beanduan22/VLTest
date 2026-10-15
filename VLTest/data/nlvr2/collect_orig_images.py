import os
import shutil

# ===== COndig =====
ROOT_DIR = "."
TARGET_DIR = "all_orig_images"
ORIG_NAME = "0_orig_image.png"

os.makedirs(TARGET_DIR, exist_ok=True)

# ===== Iterate subdirectories =====
for subdir in sorted(os.listdir(ROOT_DIR)):
    subdir_path = os.path.join(ROOT_DIR, subdir)

    # Process directories only
    if not os.path.isdir(subdir_path):
        continue

    src_image = os.path.join(subdir_path, ORIG_NAME)

    # Check if 0_orig_image.png avilable
    if not os.path.exists(src_image):
        print(f"[SKIP] {subdir} : no {ORIG_NAME}")
        continue

    dst_image = os.path.join(TARGET_DIR, f"{subdir}.png")

    shutil.copy2(src_image, dst_image)
    print(f"[OK] {src_image} -> {dst_image}")

print("Done.")
