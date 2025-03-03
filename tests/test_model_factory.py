"""
Unit tests for the model factory module.
"""

import pytest
from modules.detection import create_detector, YOLODetector, CLIPDetector


class TestModelFactory:
    """Test suite for the model factory."""

    def test_create_yolo_detector(self):
        """Test that a YOLO detector can be created correctly."""
        detector = create_detector(model_type="yolo")
        assert isinstance(detector, YOLODetector)
        
        # Test with custom parameters
        custom_detector = create_detector(
            model_type="yolo",
            model_path="yolov8n.pt",
            confidence_threshold=0.5
        )
        assert isinstance(custom_detector, YOLODetector)
        assert custom_detector.model_path == "yolov8n.pt"
        assert custom_detector.confidence_threshold == 0.5

    def test_create_clip_detector(self):
        """Test that a CLIP detector can be created correctly."""
        detector = create_detector(model_type="clip")
        assert isinstance(detector, CLIPDetector)
        
        # Test with custom parameters
        custom_categories = ["person", "dog", "cat"]
        custom_detector = create_detector(
            model_type="clip",
            categories=custom_categories,
            confidence_threshold=0.6
        )
        assert isinstance(custom_detector, CLIPDetector)
        assert custom_detector.categories == custom_categories
        assert custom_detector.confidence_threshold == 0.6

    def test_create_invalid_detector(self):
        """Test that an invalid detector type raises a ValueError."""
        with pytest.raises(ValueError):
            create_detector(model_type="invalid_model_type")

    def test_case_insensitive_yolo_lowercase(self):
        """Test that 'yolo' is case insensitive."""
        detector = create_detector(model_type="yolo")
        assert isinstance(detector, YOLODetector)
        
    def test_case_insensitive_yolo_uppercase(self):
        """Test that 'YOLO' is case insensitive."""
        detector = create_detector(model_type="YOLO")
        assert isinstance(detector, YOLODetector)
        
    def test_case_insensitive_clip_lowercase(self):
        """Test that 'clip' is case insensitive."""
        detector = create_detector(model_type="clip")
        assert isinstance(detector, CLIPDetector)
        
    def test_case_insensitive_clip_uppercase(self):
        """Test that 'CLIP' is case insensitive."""
        detector = create_detector(model_type="CLIP")
        assert isinstance(detector, CLIPDetector)