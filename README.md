# YOLO Object Detection Web Application

This application uses YOLOv8 to detect objects from your webcam in real-time and displays the results through an interactive web interface.

## Features

- Real-time object detection using webcam input
- Visual bounding boxes with class labels and confidence scores
- Interactive interface to capture and save misrecognized objects
- Ability to browse and relabel saved detections
- Performance statistics display (FPS, processing time)
- Graceful server shutdown via web UI
- Comprehensive test suite

## Requirements

- Python 3.9+
- Webcam
- Required Python packages are listed in `requirements.txt`

## Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. For frontend testing, install Playwright browsers:
   ```bash
   python -m playwright install chromium
   ```
5. Download the YOLOv8 model (happens automatically on first run)

## Usage

### Running the Server

Start the server with:
```bash
./run_server.sh
```

This will start the Flask application on http://localhost:5000

### Stopping the Server

There are two ways to stop the server:
1. Click the "Quit Server" button in the web interface
2. Run the stop script:
   ```bash
   ./stop_server.sh
   ```

### Using the Application

1. Open a web browser and navigate to http://localhost:5000
2. Allow camera access when prompted
3. Objects will be detected in real-time with bounding boxes
4. Click on an area of the video to capture and save a crop
5. Navigate to "View Saved Detections" to see all saved images
6. Relabel saved detections as needed

## Testing

### Integration Tests

Run integration tests against a running server:
```bash
python test_integration.py
```

### Frontend Tests

Test the user interface with Playwright:
```bash
pytest test_frontend.py -v
```

## How It Works

- OpenCV is used to capture frames from your webcam
- YOLOv8 processes these frames to detect objects
- Flask provides the web server to stream the processed video
- Threading is used to manage camera capture and inference simultaneously
- Canvas operations in the browser allow for user interaction with the video feed

## Technical Details

- Multi-threaded approach to separate capture, processing, and streaming
- Queues are used to safely pass data between threads
- HTML5 Canvas API for frontend image manipulation
- Graceful shutdown handling to release camera resources
- Frontend tested with Playwright for browser automation