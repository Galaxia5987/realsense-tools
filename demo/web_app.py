import streamlit as st
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

st.title("RealSense Live 4-Channel YOLO Inference")
frame_placeholder = st.empty()
stop_button = st.button("Stop Stream")

# ==========================================
# 1. LOAD YOUR CUSTOM 4-CHANNEL YOLO MODEL
# ==========================================
# Look inside the project folder you defined in your training script:
# 'rgbd_yolo/run1/weights/best.pt'
model_path = "/home/danya/best.pt"


@st.cache_resource
def load_model(path):
    print("Loading 4-channel YOLO model...")
    # Standard Ultralytics load. It will read your modified 4-channel architecture
    # directly from the weights file.
    return YOLO(path)


model = load_model(model_path)
# ==========================================

# Initialize RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
# Matches the dimensions (640, 480) from your notebook
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while not stop_button:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_np = np.asanyarray(depth_frame.get_data())
        color_np = np.asanyarray(color_frame.get_data())

        # ==========================================
        # 2. PREPROCESS TO MATCH YOUR TRAINING DATA
        # ==========================================
        # A. Convert BGR to RGB
        true_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

        # B. Normalize Depth exactly like your notebook:
        # depth_norm = (depth / depth.max() * 255).astype(np.uint8)
        depth_expanded = np.expand_dims(depth_np, axis=-1)
        max_depth = depth_expanded.max()

        if max_depth > 0:
            depth_norm = (depth_expanded / max_depth * 255.0).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_expanded, dtype=np.uint8)

        # C. Concatenate into a 4-channel image: (480, 640, 4)
        im_4ch = np.concatenate([true_rgb, depth_norm], axis=-1)

        # D. Convert to PyTorch Tensor [1, 4, 480, 640] normalized to 0.0-1.0
        # This bypasses YOLO's internal 3-channel image checks
        tensor_img = torch.from_numpy(im_4ch).permute(2, 0, 1).float() / 255.0
        tensor_img = tensor_img.unsqueeze(0)

        # ==========================================
        # 3. RUN INFERENCE & DRAW BOXES
        # ==========================================
        # Run the tensor through the model
        results = model(tensor_img, verbose=False)

        # Extract predictions
        for box in results[0].boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label_name = model.names[cls_idx]

            # Draw on the display frame (color_np)
            cv2.rectangle(color_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                color_np,
                f"{label_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # ==========================================
        # 4. DISPLAY IN STREAMLIT
        # ==========================================
        web_display = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(web_display, channels="RGB", width="stretch")

finally:
    pipeline.stop()
