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

sabotaged_tensor = original_tensor.clone()
sabotaged_tensor[:, 3, :, :] = 0.0

with torch.no_grad():
    results_original = cpu_model(original_tensor)
    results_sabotaged = cpu_model(sabotaged_tensor)

raw_orig_tensor = (
    results_original[0]
    if isinstance(results_original, (list, tuple))
    else results_original
)
raw_sabo_tensor = (
    results_sabotaged[0]
    if isinstance(results_sabotaged, (list, tuple))
    else results_sabotaged
)

abs_diff = torch.abs(raw_orig_tensor - raw_sabo_tensor)
max_diff = torch.max(abs_diff).item()
mean_diff = torch.mean(abs_diff).item()

print("\n" + "=" * 50)
print("RAW LAYER OUTPUT DIFFERENCES")
print("=" * 50)
print(f"Maximum absolute difference in raw logits: {max_diff:.7f}")
print(f"Mean absolute difference in raw logits:    {mean_diff:.7f}")

# Guidelines for raw difference:
# - If Max Diff < 1e-5: It is floating-point noise. The model is effectively ignoring depth.
# - If Max Diff > 1e-2: The model is reading and actively reacting to the depth map.

print("\n" + "=" * 50)
print("PREDICTION IMPACT (HIGH-LEVEL BENCHMARK)")
print("=" * 50)

res_orig = model.predict(original_tensor, verbose=False)[0]
res_sabo = model.predict(sabotaged_tensor, verbose=False)[0]

print("--- ORIGINAL BOXES (RGB + Depth) ---")
if len(res_orig.boxes) == 0:
    print("No objects detected.")
for box in res_orig.boxes:
    print(
        f"Class: {int(box.cls[0])} | Conf: {float(box.conf[0]):.4f} | Box: {[round(x, 2) for x in box.xyxy[0].tolist()]}"
    )

print("\n--- SABOTAGED BOXES (Depth Wiped Out) ---")
if len(res_sabo.boxes) == 0:
    print("No objects detected.")
for box in res_sabo.boxes:
    print(
        f"Class: {int(box.cls[0])} | Conf: {float(box.conf[0]):.4f} | Box: {[round(x, 2) for x in box.xyxy[0].tolist()]}"
    )
