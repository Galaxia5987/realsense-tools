import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from PIL import Image, ImageTk
import sys


class YOLOD435App:
    def __init__(self, model_path):
        self.stop_event = threading.Event()
        self.model = YOLO(model_path)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

        self.root = tk.Tk()
        self.root.title("YOLOv8 RealSense Detection")

        self.label = tk.Label(self.root)
        self.label.pack()

        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def update_frame(self):
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                results = self.model(frame, imgsz=640, device="cpu")[0]
                annotated_frame = results.plot()  # returns BGR image
                resized = cv2.resize(annotated_frame, (640, 480))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Get center of bounding box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        # Get depth at center (in mm)
                        if depth_frame:
                            depth = depth_frame.get_distance(cx, cy)
                            # Annotate distance on frame
                            cv2.putText(
                                rgb,
                                f"{depth:.2f}m",
                                (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA,
                            )

                pil_img = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(image=pil_img)

                self.label.config(image=photo)
                self.label.image = photo

            except Exception as e:
                print("Error in frame loop:", e)
                break

    def on_close(self):
        print("Stopping...")
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception as e:
            print("Pipeline stop failed:", e)
        self.root.destroy()
        sys.exit(0)


def select_model_path():
    root = tk.Tk()
    root.withdraw()  # hide root window
    file_path = filedialog.askopenfilename(
        title="Select YOLOv8 Model",
        filetypes=[("PyTorch YOLOv8 Model", "*.pt")]
    )
    root.destroy()
    return file_path


if __name__ == "__main__":
    model_path = select_model_path()
    if not model_path:
        messagebox.showinfo("No model selected", "Exiting.")
        sys.exit(0)

    try:
        app = YOLOD435App(model_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
