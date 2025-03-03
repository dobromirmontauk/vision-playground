# YOLO Object Detection Webcam Stream

This application uses YOLOv8 to detect objects from your webcam in real-time and displays the results through a web interface.

## Features

- Real-time webcam object detection using YOLOv8
- Visual bounding boxes with class labels and confidence scores
- Web-based interface to view the detection stream

## Requirements

- Python 3.9+
- Webcam
- Required Python packages are listed in `requirements.txt`

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Download the YOLOv8 model (happens automatically on first run)

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```
3. Grant camera access when prompted by your browser

## How It Works

- OpenCV is used to capture frames from your webcam
- YOLOv8 processes these frames to detect objects
- Flask provides the web server to stream the processed video
- Threading is used to manage camera capture and inference simultaneously

## Technical Details

- The app uses a multi-threaded approach to separate capture, processing, and streaming
- Queues are used to safely pass data between threads
- The YOLOv8 model draws bounding boxes around detected objects with class labels and confidence scores