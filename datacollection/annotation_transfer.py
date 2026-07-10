import os
import shutil

# --- CONFIGURATION ---
# The folder where all your original, unannotated .raw files are sitting
source_raw_dir = "dataset/rgbd/object_class"

# The folders from your unzipped Roboflow export
roboflow_labels_dir = "dataset/annotated/valid/labels"
roboflow_images_dir = "dataset/annotated/valid/images"

print("Starting file transfer...")

for label_file in os.listdir(roboflow_labels_dir):
    if not label_file.endswith(".txt"):
        continue

    base_name = label_file.replace(".txt", "")

    original_name = base_name.split("_png.rf.")[0]
    original_name = original_name.split(".rf.")[0]

    raw_filename = f"{original_name}.raw"
    source_raw_path = os.path.join(source_raw_dir, raw_filename)

    target_raw_path = os.path.join(roboflow_images_dir, raw_filename)

    if os.path.exists(source_raw_path):
        shutil.copy(source_raw_path, target_raw_path)

        png_path = os.path.join(roboflow_images_dir, f"{base_name}.png")
        if os.path.exists(png_path):
            os.remove(png_path)

        old_label_path = os.path.join(roboflow_labels_dir, label_file)
        new_label_path = os.path.join(roboflow_labels_dir, f"{original_name}.txt")
        os.rename(old_label_path, new_label_path)
    else:
        print(f"Warning: Could not find original raw file for {label_file}")

print(
    "Transfer complete! Your annotated/train/images/ folder should now contain .raw files."
)
