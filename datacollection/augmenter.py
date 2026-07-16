import os
import glob
import random
import shutil

def augment_with_depth_only(image_dir, label_dir, fraction=0.25):
    raw_files = glob.glob(os.path.join(image_dir, "*.raw"))

    original_files = [f for f in raw_files if "_depth_only" not in f]

    if not original_files:
        print(f"No original .raw files found in {image_dir}")
        return

    num_to_select = int(len(original_files) * fraction)
    selected_files = random.sample(original_files, num_to_select)

    print(f"Processing {num_to_select} files in {image_dir}...")

    rgb_bytes_len = 640 * 480 * 3
    zero_bytes = b"\x00" * rgb_bytes_len

    for raw_path in selected_files:
        base_name = os.path.splitext(os.path.basename(raw_path))[0]

        new_raw_path = os.path.join(image_dir, f"{base_name}_depth_only.raw")
        old_label_path = os.path.join(label_dir, f"{base_name}.txt")
        new_label_path = os.path.join(label_dir, f"{base_name}_depth_only.txt")

        with open(raw_path, "rb") as f:
            original_data = f.read()

        if len(original_data) <= rgb_bytes_len:
            print(f"Skipping {base_name}: File too small.")
            continue

        depth_data = original_data[rgb_bytes_len:]

        with open(new_raw_path, "wb") as f:
            f.write(zero_bytes)
            f.write(depth_data)

        if os.path.exists(old_label_path):
            shutil.copy2(old_label_path, new_label_path)


# ==========================================
# FOLDER CONFIGURATION
# ==========================================
directories = [
    ("annotated/valid/images", "annotated/valid/labels"),
]

for img_dir, lbl_dir in directories:
    if os.path.exists(img_dir) and os.path.exists(lbl_dir):
        augment_with_depth_only(img_dir, lbl_dir, fraction=0.25)
    else:
        print(f"Could not find {img_dir} or {lbl_dir}. Skipping.")

print("Done! Your dataset now has depth-only augmented examples.")
