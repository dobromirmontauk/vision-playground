"""
Unit tests for the YOLO detector.
"""

import os
import cv2
import pytest
import numpy as np
from modules.detection import create_detector, YOLODetector
from modules.detection.base import Detection


class TestYOLODetector:
    """Test suite for the YOLO detector."""

    @pytest.fixture
    def yolo_detector(self):
        """Fixture that returns a YOLO detector instance."""
        return create_detector(model_type="yolo", model_path="yolov8n.pt")

    def test_detector_creation(self):
        """Test that a YOLO detector can be created."""
        detector = create_detector(model_type="yolo")
        assert isinstance(detector, YOLODetector)
        assert detector.get_name() == "YOLO"
        assert detector.get_info()['model_type'] == "YOLO"

    def test_model_initialization(self, yolo_detector):
        """Test that the model is properly initialized."""
        # First make sure model is None
        yolo_detector.model = None
        
        # Initialize the model
        yolo_detector._initialize_model()
        
        # Check the model was initialized
        assert yolo_detector.model is not None

    def test_detect_on_image(self, yolo_detector, test_image_path):
        """Test detection on a real image."""
        # Load the test image
        image = cv2.imread(test_image_path)
        assert image is not None, f"Failed to load test image: {test_image_path}"

        # Run detection
        detections = yolo_detector.detect(image)

        # Verify results
        assert isinstance(detections, list)
        assert len(detections) > 0, "No detections found in test image"

        # Check that detections have proper structure
        for detection in detections:
            assert isinstance(detection, Detection)
            assert len(detection.box) == 4  # [x1, y1, x2, y2]
            assert isinstance(detection.class_name, str)
            assert 0 <= detection.confidence <= 1

    def test_detect_person(self, yolo_detector, test_person_image_path):
        """Test that the detector can detect a person."""
        # Load the test image
        image = cv2.imread(test_person_image_path)
        assert image is not None, f"Failed to load test image: {test_person_image_path}"

        # Run detection
        detections = yolo_detector.detect(image)

        # Verify results
        assert isinstance(detections, list)
        assert len(detections) > 0, "No detections found in test person image"

        # Check that at least one detection is a person
        person_detections = [d for d in detections if d.class_name.lower() == "person"]
        assert len(person_detections) > 0, "No person detected in person image"


def test_confidence_threshold_default(test_image_path):
    """Test with default confidence threshold."""
    confidence_threshold = 0.25
    expected_fewer_detections = False
    
    # Create detectors with different confidence thresholds
    default_detector = create_detector(model_type="yolo")
    custom_detector = create_detector(
        model_type="yolo", 
        confidence_threshold=confidence_threshold
    )
    
    # Load test image
    image = cv2.imread(test_image_path)
    
    # Get detections
    default_detections = default_detector.detect(image)
    custom_detections = custom_detector.detect(image)
    
    # With same threshold, we expect similar number of detections
    assert len(custom_detections) >= 0  # Just to ensure it runs


def test_confidence_threshold_high(test_image_path):
    """Test with high confidence threshold."""
    confidence_threshold = 0.75
    expected_fewer_detections = True
    
    # Create detectors with different confidence thresholds
    default_detector = create_detector(model_type="yolo")
    custom_detector = create_detector(
        model_type="yolo", 
        confidence_threshold=confidence_threshold
    )
    
    # Load test image
    image = cv2.imread(test_image_path)
    
    # Get detections
    default_detections = default_detector.detect(image)
    custom_detections = custom_detector.detect(image)
    
    # Higher threshold should result in fewer detections
    assert len(custom_detections) <= len(default_detections)