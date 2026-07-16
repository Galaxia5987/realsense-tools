import os
import cv2
import numpy as np

# --- CONFIGURATION ---
# 1. Dimensions of your camera's RAW files (what is on disk)
RAW_H, RAW_W = 480, 640

# 2. Dimensions expected by your YOLO model
MODEL_H, MODEL_W = 640, 640

DEPTH_DTYPE = np.uint16

stitched_raw_dir = "./qnn_quantizing_dataset/images"
qnn_output_dir = "./qnn_calibration_inputs"
os.makedirs(qnn_output_dir, exist_ok=True)

rgb_byte_size = RAW_H * RAW_W * 3  # uint8 is 1 byte
depth_element_size = np.dtype(DEPTH_DTYPE).itemsize
depth_byte_size = RAW_H * RAW_W * depth_element_size
expected_total_size = rgb_byte_size + depth_byte_size

qnn_paths = []

print("Processing files...")
for filename in os.listdir(stitched_raw_dir):
    if filename.endswith(".raw"):
        file_path = os.path.join(stitched_raw_dir, filename)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if len(file_bytes) != expected_total_size:
            print(
                f"Skipping {filename}: size is {len(file_bytes)} bytes, expected {expected_total_size}"
            )
            continue

        # Split and reconstruct to the CAMERA's original shape
        rgb_bytes = file_bytes[:rgb_byte_size]
        depth_bytes = file_bytes[rgb_byte_size:]

        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((RAW_H, RAW_W, 3))
        depth = np.frombuffer(depth_bytes, dtype=DEPTH_DTYPE).reshape((RAW_H, RAW_W, 1))

        # Apply Dynamic Depth Normalization (Replicates training loader exactly)
        depth_max = depth.max() if depth.max() > 0 else 1
        depth_norm = (depth / depth_max * 255).astype(np.uint8)

        # Concatenate into RGBD raw image (480, 640, 4) in uint8
        rgbd_raw = np.concatenate([rgb, depth_norm], axis=-1)

        # Resize the 4-channel image to the YOLO model's input size (640x640)
        rgbd_resized = cv2.resize(rgbd_raw, (MODEL_W, MODEL_H))

        # Apply the [0.0, 1.0] normalization (Replicates YOLO's internal float scaling)
        rgbd_normalized = rgbd_resized.astype(np.float32) / 255.0

        # Transpose to (4, 640, 640) and expand to (1, 4, 640, 640)
        rgbd = np.transpose(rgbd_normalized, (2, 0, 1))
        rgbd = np.expand_dims(rgbd, axis=0)

        # Save as flat float32 bytes
        out_name = f"qnn_{filename}"
        out_path = os.path.abspath(os.path.join(qnn_output_dir, out_name))
        rgbd.tofile(out_path)
        qnn_paths.append(out_path)

with open("input_list.txt", "w") as f:
    for path in qnn_paths:
        f.write(f"{path}\n")

print(f"\nSuccess! Preprocessed {len(qnn_paths)} files.")

