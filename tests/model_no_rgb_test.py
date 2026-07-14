import glob
import torch
import numpy as np
from ultralytics import YOLO

print("Loading trained model...")
model = YOLO("/home/danya/best.pt")
cpu_model = model.model.to("cpu")
cpu_model.eval()

raw_images = glob.glob("../datacollection/dataset/annotated/test/images/*.raw")
if not raw_images:
    raise FileNotFoundError("Could not find any .raw files to test with!")

test_file = raw_images[0]
print(f"Testing with file: {test_file}")

width, height = 640, 480
rgb_bytes_len = width * height * 3

with open(test_file, "rb") as f:
    raw_data = f.read()

rgb = np.frombuffer(raw_data[:rgb_bytes_len], dtype=np.uint8).reshape(
    (height, width, 3)
)
depth = np.frombuffer(raw_data[rgb_bytes_len:], dtype=np.uint16).reshape(
    (height, width, 1)
)
depth_norm = (depth / (depth.max() if depth.max() > 0 else 1) * 255).astype(np.uint8)

im = np.concatenate([rgb, depth_norm], axis=-1)
im_float = im.astype(np.float32) / 255.0
im_transposed = np.transpose(im_float, (2, 0, 1))

original_tensor = torch.from_numpy(im_transposed).unsqueeze(0).to("cpu")

blind_rgb_tensor = original_tensor.clone()
blind_rgb_tensor[:, :3, :, :] = 0.0

print("\n" + "=" * 50)
print("PREDICTION IMPACT: THE BLIND RGB TEST")
print("=" * 50)

res_orig = model.predict(original_tensor, verbose=False)[0]
res_blind = model.predict(blind_rgb_tensor, verbose=False)[0]

print("--- ORIGINAL BOXES (RGB + Depth) ---")
if len(res_orig.boxes) == 0:
    print("No objects detected.")
for box in res_orig.boxes:
    print(
        f"Class: {int(box.cls[0])} | Conf: {float(box.conf[0]):.4f} | Box: {[round(x, 2) for x in box.xyxy[0].tolist()]}"
    )

print("\n--- BLIND RGB BOXES (RGB Wiped Out, ONLY Depth Remains) ---")
if len(res_blind.boxes) == 0:
    print("No objects detected. The model is entirely dependent on RGB.")
for box in res_blind.boxes:
    print(
        f"Class: {int(box.cls[0])} | Conf: {float(box.conf[0]):.4f} | Box: {[round(x, 2) for x in box.xyxy[0].tolist()]}"
    )
