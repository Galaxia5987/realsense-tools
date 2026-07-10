import argparse
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

    base_name = os.path.splitext(os.path.basename(raw_file_path))[0]

    rgb_bytes_size = width * height * 3
    with open(raw_file_path, "rb") as f:
        raw_data = f.read()

    depth_bytes = raw_data[rgb_bytes_size:]
    depth_array = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((height, width))

    ml_depth_path = os.path.join(output_dir, f"{base_name}_depth_16bit.png")
    cv2.imwrite(ml_depth_path, depth_array)
    print(f"Saved ML-ready depth map: {ml_depth_path}")

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
    for i in range(start_idx, end_idx + 1):
        filename = f"{prefix}{i:04d}.raw"
        target_raw_file = os.path.join(base_dir, filename)

        print(f"Processing index {i}...")
        extract_depth_to_images(target_raw_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract depth images from a range of raw RGB-D files."
    )

    parser.add_argument(
        "start_idx", type=int, help="The starting index of the files to process"
    )
    parser.add_argument(
        "end_idx", type=int, help="The ending index (inclusive) of the files"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="dataset/rgbd/object_class",
        help="Directory where the .raw files are located (default: dataset/rgbd/object_class)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="object_class_",
        help="Filename prefix preceding the numerical index (default: object_class_)",
    )

    args = parser.parse_args()

    extract_range_of_depth(
        base_dir=args.dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prefix=args.prefix,
    )
