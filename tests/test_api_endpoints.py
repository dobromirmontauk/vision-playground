"""
Tests for the API endpoints of the application.
"""

import os
import json
import pytest
import threading
import queue
import numpy as np
import base64
import tempfile
from unittest.mock import MagicMock, patch
from io import BytesIO
from flask import Flask

from modules.api.routes import register_routes
from modules.utils.frame_processor import FrameData
from modules.detection.base import Detection


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture
    def app(self):
        """Fixture that returns a Flask app with registered API routes."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        app = Flask(__name__, 
                  template_folder=os.path.join(project_root, 'templates'),
                  static_folder=os.path.join(project_root, 'static'))
        app.config['TESTING'] = True
        
        # Create mock frame queue
        frame_queue = queue.Queue(maxsize=1)
        
        # Create mock frame data with detections
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [Detection([10, 20, 30, 40], "test_class", 0.9)]
        frame_data = FrameData(frame, detections)
        
        # Add annotations
        frame_data.annotate_frame(stats_data={'fps': 30, 'median_time': 10, 'max_time': 20})
        frame_data.performance_stats = {'fps': 30, 'median_time': 10, 'max_time': 20}
        
        # Put frame data in queue
        frame_queue.put(frame_data)
        
        # Create mock shutdown event
        shutdown_event = threading.Event()
        
        # Create mock stats provider
        class MockStatsProvider:
            def get_stats(self):
                return {'fps': 30, 'median_time': 10, 'max_time': 20}
        
        # Register API routes
        register_routes(app, frame_queue, shutdown_event, MockStatsProvider())
        
        # Create temporary directory for saved boxes
        os.makedirs('static/saved_boxes', exist_ok=True)
        
        return app

    def test_index_route(self, app):
        """Test the index route."""
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200

    def test_stats_endpoint(self, app):
        """Test the stats API endpoint."""
        with app.test_client() as client:
            response = client.get('/api/stats')
            assert response.status_code == 200
            
            # Check response data
            data = json.loads(response.data)
            assert 'fps' in data
            assert 'median_time' in data
            assert 'max_time' in data

    def test_frame_data_endpoint(self, app):
        """Test the frame data API endpoint."""
        with app.test_client() as client:
            response = client.get('/api/frame_data')
            assert response.status_code == 200
            
            # Check response data
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'frame' in data
            assert 'dimensions' in data
            assert 'detections' in data
            assert len(data['detections']) == 1
            
            # Check detection data
            detection = data['detections'][0]
            assert 'box' in detection
            assert 'class' in detection
            assert 'confidence' in detection

    @patch('modules.api.routes.cv2.imwrite')
    def test_save_crop_endpoint(self, mock_imwrite, app):
        """Test the save crop API endpoint."""
        with app.test_client() as client:
            # Create test data
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', test_image)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Create test metadata
            metadata = {
                'x': 10,
                'y': 20,
                'width': 30,
                'height': 40,
                'class': 'test_class',
                'confidence': 0.9
            }
            
            # Make request
            response = client.post(
                '/api/save_crop',
                json={
                    'image': f"data:image/jpeg;base64,{image_data}",
                    'metadata': metadata
                }
            )
            
            # Check response
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'filename' in data

    def test_saved_detections_route(self, app):
        """Test the saved detections route."""
        # Create a temporary test file
        test_metadata = {
            'original_class': 'test_class',
            'confidence': 0.9,
            'timestamp': '20250101_000000_000000',
            'x': 10,
            'y': 20,
            'width': 30,
            'height': 40,
            'new_label': None,
            'filename': '20250101_000000_000000_test_class_0.9.jpg'
        }
        
        metadata_path = os.path.join('static/saved_boxes', '20250101_000000_000000.json')
        with open(metadata_path, 'w') as f:
            json.dump(test_metadata, f)
        
        # Create a dummy image file
        image_path = os.path.join('static/saved_boxes', '20250101_000000_000000_test_class_0.9.jpg')
        with open(image_path, 'wb') as f:
            f.write(b'dummy image data')
        
        try:
            with app.test_client() as client:
                response = client.get('/saved_detections')
                assert response.status_code == 200
        finally:
            # Clean up test files
            os.remove(metadata_path)
            os.remove(image_path)

import cv2  # Import cv2 here for the patch to work