import os
import cv2
import numpy as np


def extract_depth_to_images(
    raw_file_path, output_dir="dataset/depth_images", width=640, height=480
):
    """Extracts depth data from a single .raw file and saves it as both a 16-bit

    exact PNG and an 8-bit colorized visualization image.
    """
    if not os.path.exists(raw_file_path):
        print(f"Warning: {raw_file_path} not found. Skipping...")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get the base name of the file (e.g., "object_class_0000")
    base_name = os.path.splitext(os.path.basename(raw_file_path))[0]

    # 1. Read the raw file bytes
    rgb_bytes_size = width * height * 3
    with open(raw_file_path, "rb") as f:
        raw_data = f.read()

    # 2. Slice out just the depth bytes (which come after the RGB bytes)
    depth_bytes = raw_data[rgb_bytes_size:]
    depth_array = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((height, width))

    # --- SAVE METHOD 1: 16-bit Grayscale PNG (For ML Pipelines) ---
    ml_depth_path = os.path.join(output_dir, f"{base_name}_depth_16bit.png")
    cv2.imwrite(ml_depth_path, depth_array)
    print(f"Saved ML-ready depth map: {ml_depth_path}")

    # --- SAVE METHOD 2: 8-bit Colorized (For Human Eyes) ---
    depth_normalized = cv2.normalize(
        depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    vis_depth_path = os.path.join(output_dir, f"{base_name}_depth_vis.png")
    cv2.imwrite(vis_depth_path, depth_colored)
    print(f"Saved visualization depth map: {vis_depth_path}\n")


def extract_range_of_depth(base_dir, start_idx, end_idx, prefix="object_class_"):
    """Loops through a range of indices (inclusive) to process multiple .raw

    files.
    """
    # end_idx + 1 makes the range inclusive of the last number
    for i in range(start_idx, end_idx + 1):
        # :04d pads the integer with leading zeros to match your 4-digit format (e.g., 0012)
        filename = f"{prefix}{i:04d}.raw"
        target_raw_file = os.path.join(base_dir, filename)

        print(f"Processing index {i}...")
        extract_depth_to_images(target_raw_file)


# ==========================================
# Run the Extraction
# ==========================================
if __name__ == "__main__":
    # The directory where your .raw files live
    source_directory = "dataset/rgbd/object_class"

    # Define your index range here (e.g., 0 to 12 inclusive)
    start_index = 0
    end_index = 12

    extract_range_of_depth(
        source_directory, start_index, end_index, prefix="object_class_"
    )
