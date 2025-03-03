"""
Unit tests for the CLIP detector.
"""

import os
import cv2
import pytest
import numpy as np
from modules.detection import create_detector, CLIPDetector
from modules.detection.base import Detection


class TestCLIPDetector:
    """Test suite for the CLIP detector."""

    @pytest.fixture
    def clip_detector(self):
        """Fixture that returns a CLIP detector instance."""
        return create_detector(model_type="clip")

    def test_detector_creation(self):
        """Test that a CLIP detector can be created."""
        detector = create_detector(model_type="clip")
        assert isinstance(detector, CLIPDetector)
        assert detector.get_name() == "CLIP-detector"
        assert detector.get_info()['model_type'] == "CLIP-detector"

    def test_detector_creation_with_custom_categories(self):
        """Test that a CLIP detector can be created with custom categories."""
        custom_categories = ["person", "dog", "cat", "car"]
        detector = create_detector(model_type="clip", categories=custom_categories)
        assert detector.categories == custom_categories
        assert len(detector.categories) == 4

    def test_model_initialization(self, clip_detector):
        """Test that models are properly initialized."""
        # Access private attributes for testing
        clip_detector._initialize_models()
        assert clip_detector.clip_model is not None
        assert clip_detector.detector_model is not None
        assert clip_detector.clip_preprocess is not None
        assert clip_detector.text_features is not None
        assert clip_detector.text_features.shape[0] == len(clip_detector.categories)

    def test_detect_on_image(self, clip_detector, test_image_path):
        """Test detection on a real image."""
        # Load the test image
        image = cv2.imread(test_image_path)
        assert image is not None, f"Failed to load test image: {test_image_path}"

        # Run detection
        detections = clip_detector.detect(image)

        # Verify results
        assert isinstance(detections, list)
        assert len(detections) > 0, "No detections found in test image"

        # Check that detections have proper structure
        for detection in detections:
            assert isinstance(detection, Detection)
            assert len(detection.box) == 4  # [x1, y1, x2, y2]
            assert isinstance(detection.class_name, str)
            assert 0 <= detection.confidence <= 1

    def test_detect_person(self, clip_detector, test_person_image_path):
        """Test that the detector can detect a person."""
        # Load the test image
        image = cv2.imread(test_person_image_path)
        assert image is not None, f"Failed to load test image: {test_person_image_path}"

        # Run detection
        detections = clip_detector.detect(image)

        # Verify results
        assert isinstance(detections, list)
        assert len(detections) > 0, "No detections found in test person image"

        # Check that at least one detection is a person
        person_detections = [d for d in detections if d.class_name.lower() == "person"]
        assert len(person_detections) > 0, "No person detected in person image"


def test_detection_count_standard_image(test_image_path):
    """Test detection count on standard test image."""
    detector = create_detector(model_type="clip")
    image = cv2.imread(test_image_path)
    detections = detector.detect(image)
    assert len(detections) >= 5, f"Expected at least 5 detections, got {len(detections)}"


def test_detection_count_person_image(test_person_image_path):
    """Test detection count on person test image."""
    detector = create_detector(model_type="clip")
    image = cv2.imread(test_person_image_path)
    detections = detector.detect(image)
    assert len(detections) >= 5, f"Expected at least 5 detections, got {len(detections)}"