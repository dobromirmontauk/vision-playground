"""
API routes for the application.
"""

import os
import time
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Response, render_template, request, jsonify, redirect, url_for
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

from ..utils import FrameData
from .utils import save_detection


def register_routes(app, frame_queue, shutdown_event, stats_provider):
    """
    Register API routes with the Flask application.
    
    Args:
        app: Flask application
        frame_queue: Queue containing processed frames
        shutdown_event: Event to signal shutdown
        stats_provider: Object that provides performance statistics
    """
    
    @app.route('/')
    def index():
        """Render the home page."""
        # Get detector info from frame processor to display in template
        detector_info = {}
        model_name = "Object Detection"
        
        if frame_queue and not frame_queue.empty():
            try:
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.put(frame_data)  # Put it back for other consumers
                
                if hasattr(frame_data, 'detector') and frame_data.detector:
                    model_name = frame_data.detector.get_name()
                    detector_info = frame_data.detector.get_info()
            except Exception as e:
                print(f"Error getting detector info: {e}")
        
        return render_template(
            'index.html', 
            model_name=model_name,
            model_info=detector_info
        )
    
    @app.route('/video_feed')
    def video_feed():
        """Route for the video feed."""
        return Response(generate_frames(frame_queue, shutdown_event),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/api/stats')
    def get_stats():
        """Return current performance stats."""
        stats = stats_provider.get_stats()
        return jsonify(stats)
    
    @app.route('/api/frame_data')
    def get_frame_data():
        """Return current frame data including detections and metadata for the frontend."""
        try:
            # Get the most recent processed frame data
            try:
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.put(frame_data)  # Put it back for other consumers
                
                # Extract original frame, detections
                original_frame = frame_data.original_frame
                detections = [d.to_dict() for d in frame_data.detections]
                
                # Create a unique frame ID based on timestamp
                frame_id = str(int(time.time() * 1000))
                
                # Encode the original frame (without annotations) as base64
                _, buffer = cv2.imencode('.jpg', original_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                # Get frame dimensions
                frame_height, frame_width = original_frame.shape[:2]
                
                return jsonify({
                    'success': True,
                    'frame_id': frame_id,
                    'frame': f"data:image/jpeg;base64,{base64_frame}",
                    'dimensions': {
                        'width': frame_width,
                        'height': frame_height
                    },
                    'detections': detections
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error getting frame: {str(e)}',
                    'detections': []
                })
        except Exception as e:
            print(f"Error getting frame data: {e}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'message': str(e),
                'detections': []
            })
    
    @app.route('/api/shutdown', methods=['POST'])
    def shutdown_server():
        """Endpoint to shut down the server."""
        try:
            print("Shutdown requested via web interface")
            
            # Set the shutdown event flag first to begin graceful shutdown process
            shutdown_event.set()
            print("Set shutdown event - threads will begin to exit")
            
            # Get the werkzeug shutdown function from the request context
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
                
                # Forcefully exit the process
                os._exit(0)
            
            # Start background thread
            import threading
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
    def save_crop_api():
        """Endpoint to save a crop from the frontend."""
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
        """Page to view saved detections."""
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
        """Update the label for a saved detection."""
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
        """Endpoint to fix a detection based on ID."""
        try:
            data = request.json
            detection_id = data.get('id')
            
            # Get the most recent processed frame
            try:
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.put(frame_data)  # Put it back for other consumers
                
                # Check if the requested detection ID exists
                if detection_id < len(frame_data.detections):
                    detection = frame_data.detections[detection_id].to_dict()
                    filename = save_detection(frame_data.original_frame, detection)
                    
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
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error accessing frame: {str(e)}'
                })
                
        except Exception as e:
            print(f"Error fixing detection: {e}")
            import traceback
            print(traceback.format_exc())
            return jsonify({'success': False, 'message': str(e)})


def generate_frames(frame_queue, shutdown_event):
    """Generator function to yield processed frames as JPEG images."""
    while not shutdown_event.is_set():
        try:
            # Use a shorter timeout during shutdown
            timeout = 0.5 if shutdown_event.is_set() else 1.0
            
            # Get the latest processed frame
            try:
                frame_data = frame_queue.get(timeout=timeout)
                frame_queue.put(frame_data)  # Put it back for other consumers
            except Exception:
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
                
            # Use the original frame (without annotations)
            frame = frame_data.original_frame
            
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