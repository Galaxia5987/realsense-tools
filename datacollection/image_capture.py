import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageTk


class RealSenseCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.width, self.height = 640, 480
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, 30
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, 30
        )

        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            messagebox.showerror(
                "Hardware Error", f"Could not start RealSense camera:\n{e}"
            )
            self.window.destroy()
            return

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # ui
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.control_frame = tk.Frame(window)
        self.control_frame.pack(pady=10)

        self.lbl_category = tk.Label(
            self.control_frame, text="Category:", font=("Arial", 12)
        )
        self.lbl_category.pack(side=tk.LEFT, padx=5)

        self.entry_category = tk.Entry(self.control_frame, font=("Arial", 12))
        self.entry_category.insert(0, "object_class")
        self.entry_category.pack(side=tk.LEFT, padx=5)

        self.btn_snapshot = tk.Button(
            self.control_frame,
            text="take snapshot",
            bg="lightgreen",
            font=("Arial", 12, "bold"),
            command=self.snapshot,
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=10)

        self.window.bind("<space>", self.snapshot)

        self.base_folder = "dataset"
        self.image_count = 0
        self.current_color_frame = None
        self.current_depth_frame = None

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if depth_frame and color_frame:
                self.current_depth_frame = np.asanyarray(depth_frame.get_data())
                self.current_color_frame = np.asanyarray(color_frame.get_data())

                rgb_display = cv2.cvtColor(self.current_color_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_display)
                self.photo = ImageTk.PhotoImage(image=pil_img)

                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        except Exception as e:
            print(f"Frame capture error: {e}")

        self.window.after(15, self.update_frame)

    def snapshot(self, event=None):
        if self.current_color_frame is None or self.current_depth_frame is None:
            messagebox.showwarning("Warning", "Camera stream not ready.")
            return

        category = self.entry_category.get().strip()
        if not category:
            messagebox.showwarning("Warning", "Please input a category name.")
            return

        rgb_dir = os.path.join(self.base_folder, "rgb", category)
        rgbd_dir = os.path.join(self.base_folder, "rgbd", category)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(rgbd_dir, exist_ok=True)

        base_name = f"{category}_{self.image_count:04d}"

        png_path = os.path.join(rgb_dir, f"{base_name}.png")
        cv2.imwrite(png_path, self.current_color_frame)

        true_rgb = cv2.cvtColor(self.current_color_frame, cv2.COLOR_BGR2RGB)
        raw_path = os.path.join(rgbd_dir, f"{base_name}.raw")

        with open(raw_path, "wb") as f:
            f.write(true_rgb.tobytes())
            f.write(self.current_depth_frame.tobytes())

        print(f"Saved: {png_path} AND {raw_path}")
        self.image_count += 1

    def on_closing(self):
        self.pipeline.stop()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseCollectorApp(root, "RealSense RGBD Dataset Collector")
    root.mainloop()
