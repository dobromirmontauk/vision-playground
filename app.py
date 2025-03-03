#!/usr/bin/env python3
"""
Main application file for the Object Detection web app.

This application allows users to:
1. View real-time object detection from a webcam
2. Save and review detected objects
3. Fix mislabeled objects

Supported detection models:
- YOLO: Fast object detection using YOLOv8
- CLIP: Zero-shot detection using CLIP for classification
"""

import os
import sys
import time
import signal
import threading
import queue
import atexit
import argparse
from flask import Flask

from modules.camera import create_camera
from modules.detection import create_detector
from modules.utils import FrameProcessor
from modules.api import register_routes

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Web App")
    parser.add_argument("--model", type=str, default="yolo", choices=["yolo", "clip"],
                      help="Detection model to use (default: yolo)")
    parser.add_argument("--confidence", type=float, default=0.25,
                      help="Confidence threshold for detections (default: 0.25)")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port to run the web server on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to run the web server on (default: 0.0.0.0)")
    parser.add_argument("--categories", type=str, nargs='+',
                      help="Custom categories for CLIP detector (only applicable with --model=clip)")
    
    return parser.parse_args()

# Initialize Flask app
app = Flask(__name__)

# Create folder for saving misrecognized objects
os.makedirs('static/saved_boxes', exist_ok=True)

# Global variables
frame_queue = queue.Queue(maxsize=1)  # queue.Queue uses maxsize, not maxlen
shutdown_event = threading.Event()
frame_processor = None  # Will be set in the capture_and_process function


def capture_and_process(args):
    """Thread function to capture and process frames from the camera."""
    # Initialize components
    camera = create_camera(use_test_camera=False, width=640, height=480, fps=30)
    
    # Create detector based on command line arguments
    detector_kwargs = {
        'confidence_threshold': args.confidence
    }
    
    # Add model-specific arguments
    if args.model.lower() == "yolo":
        detector_kwargs['model_path'] = 'yolov8n.pt'
    elif args.model.lower() == "clip":
        if args.categories:
            detector_kwargs['categories'] = args.categories
    
    # Create the detector
    detector = create_detector(model_type=args.model, **detector_kwargs)
    processor = FrameProcessor(detector)
    
    # Make the processor available globally
    global frame_processor
    frame_processor = processor
    
    consecutive_failures = 0
    max_failures = 5
    
    print(f"Starting capture with {detector.get_name()} detector")
    print(f"Detector info: {detector.get_info()}")
    
    while not shutdown_event.is_set():
        try:
            # Check if camera is still open
            if not camera.is_opened():
                print("Camera is not open in capture thread. Attempting to reinitialize...")
                camera = create_camera(use_test_camera=False)
                time.sleep(1)
                continue
                
            # Read a frame
            success, frame = camera.read()
            
            if success:
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Process the frame with timeout protection
                try:
                    # Add a timeout mechanism to prevent hanging on a single frame
                    process_timeout = 5.0  # 5 seconds max per frame
                    
                    # Create a timer to measure how long we're spending on a frame
                    start_time = time.time()
                    frame_data = processor.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    print(f"Frame processed in {processing_time:.2f} seconds")
                    
                    # If the queue is full, remove the old frame
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    # Put the new processed frame in the queue
                    try:
                        if not shutdown_event.is_set():  # Don't put new frames during shutdown
                            frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        pass
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    import traceback
                    print(traceback.format_exc())
                    # Prevent CPU spinning on errors
                    time.sleep(0.1)
            else:
                # Handle read failure
                consecutive_failures += 1
                print(f"Failed to read from camera ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Reinitializing camera...")
                    camera.release()
                    camera = create_camera(use_test_camera=False)
                    consecutive_failures = 0
                
                time.sleep(0.5)
        
        except Exception as e:
            if not shutdown_event.is_set():  # Only log errors if not shutting down
                print(f"Error in capture_and_process: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(1)
    
    # Release camera when exiting
    camera.release()
    print("Capture thread exiting cleanly")


def start_processing_thread(args):
    """Start the thread for capturing and processing frames."""
    process_thread = threading.Thread(target=capture_and_process, args=(args,), daemon=True)
    process_thread.start()
    return process_thread


def shutdown_handler():
    """Handle application shutdown."""
    print("Application shutting down...")
    
    # Signal threads to shut down
    shutdown_event.set()
    print("Notified threads to shut down, waiting for them to exit...")
    
    # Give threads a moment to exit cleanly
    time.sleep(3)
    
    print("Shutdown complete")


def signal_handler(sig, frame):
    """Handle signals like SIGTERM and SIGINT."""
    print(f"Received signal {sig}, shutting down gracefully...")
    shutdown_handler()
    sys.exit(0)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Register the shutdown handler with atexit
    atexit.register(shutdown_handler)
    
    try:
        # Start the processing thread
        process_thread = start_processing_thread(args)
        
        # We need access to the processor's performance monitor from the main thread
        class StatsProvider:
            def __init__(self):
                self.last_stats = {'fps': 0, 'median_time': 0, 'max_time': 0}
            
            def get_stats(self):
                global frame_processor
                if frame_processor is not None:
                    # Get stats directly from the frame processor
                    return frame_processor.get_stats()
                return self.last_stats
        
        # Register API routes
        register_routes(app, frame_queue, shutdown_event, StatsProvider())
        
        # Start the Flask app
        print(f"Starting web server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)