# Object Detection Web Application

This application performs real-time object detection from your webcam and displays the results through an interactive web interface. It supports multiple detection models:

- **YOLO**: Fast object detection using YOLOv8
- **CLIP**: Zero-shot object classification using OpenAI's CLIP model (supports 1000+ household item categories)

## Features

- Real-time object detection using webcam input 
- Multiple object detection models to choose from
- Visual bounding boxes with class labels and confidence scores
- Interactive interface to capture and save detections
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
# Run with default YOLO detector
./run_server.sh

# Run with CLIP detector
./run_clip_detector.sh

# Run with CLIP detector on Apple Silicon GPU (M1/M2/M3)
./run_with_mps.sh

# Or specify a detector model and options directly
python app.py --model yolo  # Use YOLO detector
python app.py --model clip  # Use CLIP detector

# Additional options
python app.py --model clip --confidence 0.4 --port 8080
python app.py --model clip --categories person car dog chair  # Specific categories for CLIP
python app.py --model clip --categories-file path/to/categories.txt  # Load categories from file
python app.py --model clip --max-categories 100  # Limit number of categories
```

This will start the Flask application on http://localhost:5000 (or custom port if specified)

### Using GPU Acceleration

On Apple Silicon Macs (M1/M2/M3), you can use GPU acceleration for the CLIP detector:

1. Ensure you have PyTorch 2.5.0+ installed with MPS support
2. Run the application with `./run_with_mps.sh`
3. Check the logs to verify that MPS acceleration is being used

This significantly improves performance for the CLIP detector.

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

The project has a comprehensive test suite including unit tests, integration tests, and frontend tests.

### Running Tests

To run the unit tests:
```bash
./run_tests.sh
```

To run all tests including integration and frontend tests:
```bash
./run_tests.sh --with-integration
```

Alternatively, you can use pytest directly:
```bash
# Run all unit tests
pytest tests/

# Run a specific test file
pytest tests/test_clip_detector.py

# Run with verbose output
pytest tests/ -v

# Run only tests with a specific marker
pytest -m "not integration" tests/
```

### Test Types

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the API endpoints with a running server
- **Frontend Tests**: Test the user interface with Playwright

## How It Works

- OpenCV is used to capture frames from your webcam
- Object detection models process frames to detect objects:
  - **YOLO**: Fast general-purpose object detection
  - **CLIP**: Zero-shot classification using natural language understanding
- Flask provides the web server to stream the processed video
- Threading is used to manage camera capture and inference simultaneously
- Canvas operations in the browser allow for user interaction with the video feed

### Testing CLIP Detector

You can test the CLIP detector independently using:
```bash
python test_clip_detector.py --model clip --categories person car dog chair
```

This will capture a frame from your webcam (or use a specified image) and test the CLIP detection on it.

## Technical Details

- Multi-threaded approach to separate capture, processing, and streaming
- Queues are used to safely pass data between threads
- HTML5 Canvas API for frontend image manipulation
- Graceful shutdown handling to release camera resources
- Frontend tested with Playwright for browser automation