# Object Detection with YOLO

## Project Description
This project implements real-time object detection using YOLO (You Only Look Once) algorithm. YOLO is a state-of-the-art, real-time object detection system that can identify multiple objects in images and video streams with high accuracy and speed.

## Features
- Real-time object detection in video streams
- Support for multiple YOLO versions (YOLOv3, YOLOv4, YOLOv5, YOLOv8)
- Detection of 80+ object classes from COCO dataset
- Bounding box visualization with confidence scores
- GPU acceleration support for faster inference

## Requirements
- Python 3.8+
- OpenCV
- PyTorch or TensorFlow
- NumPy
- CUDA (optional, for GPU acceleration)

## Usage
The main implementation will be in `yolo_detector.py`

## Dataset
- COCO dataset for pre-trained models
- Custom dataset support for fine-tuning

## Expected Output
- Detected objects with bounding boxes
- Class labels and confidence scores
- FPS metrics for performance evaluation
