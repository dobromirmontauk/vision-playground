import cv2
import time
import os
import sys
import json
import signal
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
import threading
import queue
import numpy as np
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
import collections
import contextlib

# Silence ultralytics import and other noisy modules
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    from ultralytics import YOLO

app = Flask(__name__)

# Create folder for saving misrecognized objects
os.makedirs('static/saved_boxes', exist_ok=True)

# Global variables for sharing data between threads
frame_queue = queue.Queue(maxsize=1)
processed_frame_queue = queue.Queue(maxsize=1)

# Performance monitoring
processing_times = collections.deque(maxlen=100)  # Store last 100 processing times
fps_counter = collections.deque(maxlen=50)  # Store timestamps for FPS calculation
stats = {
    'fps': 0,
    'median_time': 0,
    'max_time': 0
}

# Threading control
shutdown_event = threading.Event()

# Load YOLOv8 model (silently)
with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    model = YOLO('yolov8n.pt')  # Using the smallest model for quick startup

# Initialize camera
camera = None

def initialize_camera():
    global camera
    
    # Check if camera is already initialized and open
    if camera is not None and camera.isOpened():
        print("Camera already initialized")
        return camera
    
    print("Initializing camera...")
    camera = cv2.VideoCapture(0)  # 0 is usually the default camera
    
    # Try opening the camera a few times before giving up
    max_attempts = 3
    for attempt in range(max_attempts):
        if camera.isOpened():
            print(f"Camera initialized successfully on attempt {attempt+1}/{max_attempts}")
            
            # Configure camera for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            return camera
        else:
            print(f"Camera initialization attempt {attempt+1}/{max_attempts} failed.")
            time.sleep(1)
            camera.release()
            camera = cv2.VideoCapture(0)
    
    # If we got here, camera initialization failed
    print("ERROR: Could not access the camera after multiple attempts!")
    print("On macOS, you might need to grant camera permissions to Terminal/iTerm2.")
    print("Go to System Preferences > Security & Privacy > Privacy > Camera")
    print("Using test camera as fallback.")
    
    # Create a test camera as fallback
    return create_test_camera()

def create_test_camera():
    """Create a fake camera that produces test frames"""
    global camera
    # Close the failed camera if it exists
    if camera is not None:
        camera.release()
    
    # Create a class that mimics a cv2.VideoCapture to produce test frames
    class TestCamera:
        def __init__(self):
            self.frame_count = 0
            self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.opened = True
            
        def isOpened(self):
            return self.opened
            
        def read(self):
            # Create a test frame with a moving rectangle to simulate motion
            frame = self.test_frame.copy()
            # Draw a gray background
            frame[:] = (100, 100, 100)
            
            # Draw a moving rectangle (simulating an object to detect)
            x = 100 + int(100 * np.sin(self.frame_count / 30))
            y = 200 + int(50 * np.cos(self.frame_count / 20))
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
            
            # Draw another rectangle
            x2 = 400 + int(50 * np.sin(self.frame_count / 25))
            y2 = 300 + int(30 * np.cos(self.frame_count / 15))
            cv2.rectangle(frame, (x2, y2), (x2 + 70, y2 + 70), (0, 0, 255), -1)
            
            # Add text explaining this is a test frame
            cv2.putText(frame, "TEST MODE - No camera access", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            self.frame_count += 1
            return True, frame
            
        def release(self):
            self.opened = False
    
    camera = TestCamera()
    print("Using test camera with simulated objects")
    return camera

def capture_frames():
    """Thread function to continuously capture frames from the camera"""
    global camera
    consecutive_failures = 0
    max_failures = 5
    
    while not shutdown_event.is_set():
        try:
            # Check if camera is still open
            if camera is None or not camera.isOpened():
                print("Camera is not open in capture thread. Attempting to reinitialize...")
                camera = initialize_camera()
                time.sleep(1)
                continue
                
            # Read a frame
            success, frame = camera.read()
            
            if success:
                # Reset failure counter on success
                consecutive_failures = 0
                
                # If the queue is full, remove the old frame
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Put the new frame in the queue
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                # Handle read failure
                consecutive_failures += 1
                print(f"Failed to read from camera ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Reinitializing camera...")
                    if camera is not None:
                        camera.release()
                    camera = initialize_camera()
                    consecutive_failures = 0
                
                time.sleep(0.5)
        
        except Exception as e:
            if not shutdown_event.is_set():  # Only log errors if not shutting down
                print(f"Error in capture_frames: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(1)
    
    print("Capture thread exiting cleanly")

def process_frames():
    """Thread function to process frames with YOLO"""
    global stats
    while not shutdown_event.is_set():
        try:
            # Get the latest frame with a shorter timeout during shutdown
            timeout = 1.0 if shutdown_event.is_set() else 5.0
            try:
                frame = frame_queue.get(timeout=timeout)
            except queue.Empty:
                if shutdown_event.is_set():
                    break  # Exit immediately if we're shutting down
                # Create a placeholder frame if there's no input and we're not shutting down
                # Silently wait for a frame
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty_frame, "Waiting for camera input...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Put the empty frame in the queue so the web view has something to show
                if processed_frame_queue.full():
                    try:
                        processed_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                try:
                    if not shutdown_event.is_set():  # Don't put new frames during shutdown
                        processed_frame_queue.put_nowait((empty_frame, {'timestamp': time.time(), 'frame': empty_frame, 'detections': []}))
                except queue.Full:
                    pass
                    
                time.sleep(1.0)
                continue
            
            # Check again if we're shutting down before processing
            if shutdown_event.is_set():
                break
                
            # Record start time for processing
            start_time = time.time()
            
            # Make a copy to ensure we don't modify the original frame
            frame_copy = frame.copy()
            
            # Run YOLOv8 inference on the frame - suppress all output
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                results = model(frame_copy)
            
            # Calculate processing time in milliseconds
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            # Update FPS counter
            current_time = time.time()
            fps_counter.append(current_time)
            if len(fps_counter) > 1:
                # Calculate FPS based on timestamps in the deque
                time_diff = fps_counter[-1] - fps_counter[0]
                if time_diff > 0:
                    stats['fps'] = round(len(fps_counter) / time_diff, 1)
            
            # Update other stats
            if processing_times:
                stats['median_time'] = round(np.median(processing_times), 1)
                stats['max_time'] = round(max(processing_times), 1)
                
            # Get original detection data for interactive elements
            detections = []
            if len(results) > 0:
                # Extract boxes, classes, and confidence scores
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_names = results[0].names
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(classes[i])
                    confidence = float(confs[i])
                    class_name = class_names[class_id]
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': class_name,
                        'confidence': round(confidence, 2)
                    })
            
            # Create a copy of the frame for custom annotation
            annotated_frame = frame.copy()
            
            # Draw bounding boxes with interactive buttons
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                class_name = det['class']
                confidence = det['confidence']
                
                # Generate a unique color for this class (hash the class name)
                class_hash = hash(class_name) % 0xFFFFFF
                r = (class_hash & 0xFF0000) >> 16
                g = (class_hash & 0x00FF00) >> 8
                b = class_hash & 0x0000FF
                color = (b, g, r)  # OpenCV uses BGR
                
                # Ensure the color is visible
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                if brightness < 0.4:  # If color is too dark
                    color = (min(b + 100, 255), min(g + 100, 255), min(r + 100, 255))
                
                # Draw box with class-specific color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name} {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                
                # Choose text color (black or white) based on background brightness
                text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                
                # Draw "Fix" button - make it bigger and more visible
                button_width = 50
                button_height = 30
                button_x = x2 - button_width
                button_y = y1
                cv2.rectangle(annotated_frame, (button_x, button_y), (x2, button_y + button_height), (0, 0, 255), -1)
                cv2.putText(annotated_frame, "FIX", (button_x + 10, button_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection ID overlay (transparent)
                det_id = f"id:{i}"
                cv2.putText(annotated_frame, det_id, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add performance stats to frame
            stats_text = f"FPS: {stats['fps']} | Median: {stats['median_time']} ms | Max: {stats['max_time']} ms"
            cv2.putText(annotated_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Store frame data for API access
            frame_data = {
                'timestamp': time.time(),
                'frame': frame.copy(),
                'detections': detections
            }
            
            # If the queue is full, remove the old processed frame
            if processed_frame_queue.full():
                try:
                    processed_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Put the new processed frame and data in the queue
            try:
                if not shutdown_event.is_set():  # Don't put new frames during shutdown
                    # When we have detections, always keep the latest frame in the queue
                    # For saving with the "Fix" button
                    processed_frame_queue.put_nowait((annotated_frame, frame_data))
            except queue.Full:
                pass
                
        except queue.Empty:
            # No frame available, create a placeholder (this should be handled above now)
            pass
        except Exception as e:
            if not shutdown_event.is_set():  # Only log errors if not shutting down
                print(f"Error in process_frames: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(0.1)
    
    print("Processing thread exiting cleanly")

def generate_frames():
    """Generator function to yield processed frames as JPEG images"""
    while not shutdown_event.is_set():
        try:
            # Use a shorter timeout during shutdown
            timeout = 0.5 if shutdown_event.is_set() else 1.0
            
            # Get the latest processed frame and data
            try:
                frame_tuple = processed_frame_queue.get(timeout=timeout)
            except queue.Empty:
                if shutdown_event.is_set():
                    break  # Exit if we're shutting down
                
                # Create placeholder frame when queue is empty and we're not shutting down
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty_frame, "Waiting for camera...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', empty_frame)
                if ret and not shutdown_event.is_set():  # Skip yielding during shutdown
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.5)  # Wait a bit longer for empty frames
                continue
            
            # Check again if we're shutting down
            if shutdown_event.is_set():
                break
                
            # Process the frame
            if isinstance(frame_tuple, tuple):
                frame, _ = frame_tuple
            else:
                frame = frame_tuple  # For backwards compatibility
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Yield the frame in the format expected by Flask's Response
            if not shutdown_event.is_set():  # Skip yielding during shutdown
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                       
        except Exception as e:
            if not shutdown_event.is_set():  # Only log errors if not shutting down
                print(f"Error in generate_frames: {e}")
                import traceback
                print(traceback.format_exc())
                time.sleep(0.1)
                
    print("Frame generator exiting cleanly")

# Function to save a detected object
def save_detection(frame, detection):
    """Save a detected object to disk"""
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Extract the bounding box and crop the image
        x1, y1, x2, y2 = detection['box']
        cropped_img = frame[y1:y2, x1:x2]
        
        # Convert to RGB for PIL
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_img_rgb)
        
        # Save the cropped image
        filename = f"{timestamp}_{detection['class']}_{detection['confidence']}.jpg"
        filepath = os.path.join('static/saved_boxes', filename)
        pil_img.save(filepath)
        
        # Save metadata
        metadata = {
            'original_class': detection['class'],
            'confidence': detection['confidence'],
            'timestamp': timestamp,
            'new_label': None,
            'filename': filename
        }
        
        metadata_path = os.path.join('static/saved_boxes', f"{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return filename
    except Exception as e:
        print(f"Error saving detection: {e}")
        return None

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Return current performance stats"""
    print(f"Returning stats: {stats}")
    return jsonify(stats)

@app.route('/api/shutdown', methods=['POST'])
def shutdown_server():
    """Endpoint to shut down the server"""
    try:
        print("Shutdown requested via web interface")
        
        # Set the shutdown event flag first to begin graceful shutdown process
        shutdown_event.set()
        print("Set shutdown event - threads will begin to exit")
        
        # Get the werkzeug shutdown function from the request context
        # IMPORTANT: Need to do this in the request context, before starting the thread
        werkzeug_shutdown = None
        try:
            werkzeug_shutdown = request.environ.get('werkzeug.server.shutdown')
            if werkzeug_shutdown:
                print("Found Werkzeug shutdown function")
        except Exception as e:
            print(f"Error accessing Werkzeug shutdown: {e}")
        
        # Use a background thread to shutdown the server to allow response to be sent
        def shutdown_after_response():
            # Give time for response to be sent and for threads to start exiting
            time.sleep(2)
            print("Background thread initiating final shutdown")
            
            # First stop our threads by setting the shutdown event
            print("All threads should be exiting now")
            time.sleep(1)  # Give threads time to exit

            # Now shut down the server
            if werkzeug_shutdown:
                try:
                    print("Shutting down Werkzeug server...")
                    werkzeug_shutdown()
                    print("Werkzeug server shut down")
                except Exception as e:
                    print(f"Error shutting down Werkzeug server: {e}")
            
            # Use the standard signal-based shutdown as backup
            print("Using signal-based shutdown")
            # Ensure camera is released
            release_camera()
            # Forcefully exit the process
            os._exit(0)
                
        # Start background thread
        shutdown_thread = threading.Thread(target=shutdown_after_response)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
        return jsonify({'success': True, 'message': 'Server is shutting down...'})
    except Exception as e:
        print(f"Error during shutdown: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/save_crop', methods=['POST'])
def save_crop():
    """Endpoint to save a crop from the frontend"""
    try:
        # Get data from the request
        data = request.json
        image_data = data.get('image', '')
        metadata = data.get('metadata', {})
        
        # Log info about the received crop
        print(f"Received crop: {metadata.get('width')}x{metadata.get('height')} at position ({metadata.get('x')}, {metadata.get('y')})")
        
        # Extract the base64 image data (remove the data URL prefix)
        if image_data.startswith('data:image'):
            # Extract everything after the comma
            base64_data = image_data.split(',')[1]
        else:
            base64_data = image_data
            
        # Decode the base64 image
        image_bytes = base64.b64decode(base64_data)
        
        # Open the image using PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Filename with metadata
        class_name = metadata.get('class', 'manual_selection')
        confidence = metadata.get('confidence', 1.0)
        filename = f"{timestamp}_{class_name}_{confidence}.jpg"
        filepath = os.path.join('static/saved_boxes', filename)
        
        # Save the image
        image.save(filepath)
        
        # Save additional metadata
        metadata_obj = {
            'original_class': class_name,
            'confidence': confidence,
            'timestamp': timestamp,
            'x': metadata.get('x', 0),
            'y': metadata.get('y', 0),
            'width': metadata.get('width', 0),
            'height': metadata.get('height', 0),
            'new_label': None,
            'filename': filename
        }
        
        metadata_path = os.path.join('static/saved_boxes', f"{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_obj, f)
            
        print(f"Successfully saved crop as {filename}")
        return jsonify({
            'success': True, 
            'message': f'Crop saved as {filename}',
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error saving crop: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/saved_detections')
def saved_detections():
    """Page to view saved detections"""
    saved_items = []
    
    # Get all JSON metadata files
    json_files = [f for f in os.listdir('static/saved_boxes') if f.endswith('.json')]
    
    for json_file in sorted(json_files, reverse=True):
        try:
            with open(os.path.join('static/saved_boxes', json_file), 'r') as f:
                metadata = json.load(f)
                
            # Check if the corresponding image exists
            img_path = os.path.join('static/saved_boxes', metadata['filename'])
            if os.path.exists(img_path):
                saved_items.append(metadata)
        except Exception as e:
            print(f"Error loading metadata {json_file}: {e}")
    
    return render_template('saved_detections.html', items=saved_items)

@app.route('/api/update_label', methods=['POST'])
def update_label():
    """Update the label for a saved detection"""
    try:
        data = request.json
        filename = data.get('filename')
        new_label = data.get('new_label')
        
        # Get timestamp from filename
        timestamp = filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
        metadata_path = os.path.join('static/saved_boxes', f"{timestamp}.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['new_label'] = new_label
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'message': 'Metadata file not found'})
    except Exception as e:
        print(f"Error updating label: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/fix_detection', methods=['POST'])
def fix_detection():
    """Endpoint to fix a detection based on ID"""
    try:
        data = request.json
        detection_id = data.get('id')
        
        # Get the most recent processed frame and its detections
        try:
            frame_data = processed_frame_queue.get(timeout=1.0)
            processed_frame_queue.put(frame_data)  # Put it back for other consumers
            
            if isinstance(frame_data, tuple):
                _, frame_info = frame_data
                frame = frame_info.get('frame')
                detections = frame_info.get('detections', [])
            else:
                # For backwards compatibility
                frame = frame_data
                detections = []
            
            # Check if the requested detection ID exists
            if detection_id < len(detections):
                detection = detections[detection_id]
                filename = save_detection(frame, detection)
                
                if filename:
                    return jsonify({
                        'success': True,
                        'message': f'Detection saved as {filename}',
                        'filename': filename
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to save detection'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Detection ID not found in current frame'
                })
        except queue.Empty:
            return jsonify({
                'success': False,
                'message': 'No frame available'
            })
            
    except Exception as e:
        print(f"Error fixing detection: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

def start_processing_threads():
    """Start the threads for capturing and processing frames"""
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    
    capture_thread.start()
    process_thread.start()

def release_camera():
    """Release camera resources when shutting down"""
    global camera
    
    # Always log these messages to make sure they appear in the log file
    sys.stdout.write("Releasing camera resources...\n")
    sys.stdout.flush()
    
    if camera is not None:
        try:
            camera.release()
            sys.stdout.write("Camera resources released successfully\n")
        except Exception as e:
            sys.stdout.write(f"Error releasing camera: {e}\n")
    else:
        sys.stdout.write("No camera resources to release\n")
    
    sys.stdout.flush()

def shutdown_handler():
    """Handle application shutdown"""
    global shutdown_event
    print("Application shutting down...")
    
    # Signal threads to shut down
    shutdown_event.set()
    print("Notified threads to shut down, waiting for them to exit...")
    
    # Give threads a moment to exit cleanly
    time.sleep(3)
    
    # Now release camera resources
    release_camera()
    print("Shutdown complete")

def signal_handler(sig, frame):
    """Handle signals like SIGTERM and SIGINT"""
    print(f"Received signal {sig}, shutting down gracefully...")
    shutdown_handler()
    sys.exit(0)

if __name__ == '__main__':
    import atexit
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Register the shutdown handler with atexit
    atexit.register(shutdown_handler)
    
    try:
        # Initialize the camera
        camera = initialize_camera()
        
        # Check if we got a test camera instead of a real one
        if isinstance(camera, object) and hasattr(camera, 'isOpened') and not camera.isOpened():
            print("WARNING: Running with test camera due to camera access issues.")
            print("To force exit instead, uncomment the sys.exit(1) line in app.py")
            # Uncomment the next line to force exit if camera isn't available
            # sys.exit(1)
        
        # Start the processing threads
        start_processing_threads()
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        print(traceback.format_exc())
        release_camera()
        sys.exit(1)