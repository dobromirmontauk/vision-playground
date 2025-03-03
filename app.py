#!/usr/bin/env python3
"""
Main application file for the YOLO Object Detection web app.

This application allows users to:
1. View real-time object detection from a webcam
2. Save and review detected objects
3. Fix mislabeled objects
"""

import os
import sys
import time
import signal
import threading
import queue
import atexit
from flask import Flask

from modules.camera import create_camera
from modules.detection import create_detector
from modules.utils import FrameProcessor
from modules.api import register_routes

# Initialize Flask app
app = Flask(__name__)

# Create folder for saving misrecognized objects
os.makedirs('static/saved_boxes', exist_ok=True)

# Global variables
frame_queue = queue.Queue(maxlen=1)
shutdown_event = threading.Event()
frame_processor = None  # Will be set in the capture_and_process function


def capture_and_process():
    """Thread function to capture and process frames from the camera."""
    # Initialize components
    camera = create_camera(use_test_camera=False, width=640, height=480, fps=30)
    detector = create_detector(model_type="yolo", model_path='yolov8n.pt')
    processor = FrameProcessor(detector)
    
    # Make the processor available globally
    global frame_processor
    frame_processor = processor
    
    consecutive_failures = 0
    max_failures = 5
    
    print(f"Starting capture with {detector.get_name()} detector")
    
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
                
                # Process the frame
                frame_data = processor.process_frame(frame)
                
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


def start_processing_thread():
    """Start the thread for capturing and processing frames."""
    process_thread = threading.Thread(target=capture_and_process, daemon=True)
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
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Register the shutdown handler with atexit
    atexit.register(shutdown_handler)
    
    try:
        # Start the processing thread
        process_thread = start_processing_thread()
        
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
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)