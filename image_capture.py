import tkinter as tk
from tkinter import messagebox
import os
import cv2
import numpy as np
import pyrealsense2 as rs

IMAGE_FOLDER = "captured_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

class RealSenseCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Camera Capture")
        self.image_index = 0
        self.pipeline = None
        self.running = False

        self.preview_label = tk.Label(root)
        self.preview_label.pack()

        self.start_btn = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=5)

        self.capture_btn = tk.Button(root, text="Capture Image", command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack(pady=5)

        self.stop_btn = tk.Button(root, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        self.status = tk.Label(root, text="Camera not started")
        self.status.pack(pady=10)

    def start_camera(self):
        if self.running:
            return
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.running = True
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.start_btn.config(state=tk.DISABLED)
            self.status.config(text="Camera started")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{e}")

    def stop_camera(self):
        if self.pipeline:
            self.running = False
            self.pipeline.stop()
            self.pipeline = None
            self.capture_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.start_btn.config(state=tk.NORMAL)
            self.preview_label.config(image="")
            self.status.config(text="Camera stopped")

    def update_preview(self):
        if not self.running:
            return

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.last_frame = img  # store for capture

        img_tk = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = cv2.resize(img_tk, (320, 240))
        img_tk = cv2.cvtColor(img_tk, cv2.COLOR_RGB2BGR)
        img_tk = cv2.imencode('.png', img_tk)[1].tobytes()

        from PIL import Image, ImageTk
        photo = ImageTk.PhotoImage(data=img_tk)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

        self.root.after(30, self.update_preview)

    def capture_image(self):
        if not hasattr(self, 'last_frame'):
            return
        filename = f"image_{self.image_index:04d}.jpg"
        path = os.path.join(IMAGE_FOLDER, filename)
        cv2.imwrite(path, self.last_frame)
        self.image_index += 1
        self.status.config(text=f"Saved: {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseCaptureApp(root)
    root.mainloop()
