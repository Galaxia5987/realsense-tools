# RealSense Camera YOLO Tools

This repository contains Python scripts for working with Intel RealSense cameras, designed to streamline image capture and object detection workflows for use with tools like [Roboflow](https://roboflow.com/).

## Scripts

### 1. `image_capture.py`

A simple GUI for capturing images from a RealSense camera.

### 2. `detection_gui.py`

A GUI for running a trained YOLOv8 `.pt` model on live RealSense camera input.

## Installation

1. Clone the repository: `git clone http://github.com/Galaxia5987/realsense-tools`

2. Make a virtual enviroment and install dependencies: 
```bash
cd CHANGEME
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run one of the scripts: `python3 image_capture.py`

## Typical Workflow

1. **Capture Images:**  
     Use `image_capture.py` to collect and save images for training.

2. **Annotate:**  
     Annotate images using Roboflow.

3. **Train Model:**
     Train the model using [this](https://www.kaggle.com/code/adarwas/yolov8-traning-and-conversion-to-rknn) Kaggle notebook.

4. **Run Detection:**  
     Use `detection_gui.py` to load your trained `.pt` model and perform real-time detection from a Realsense camera.

## Notes

- Make sure your RealSense camera is connected before running the scripts.