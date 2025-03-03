"""
Unit tests for the FrameProcessor class.
"""

import cv2
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from modules.utils.frame_processor import FrameProcessor, FrameData
from modules.detection import create_detector
from modules.detection.base import Detection


class TestFrameProcessor:
    """Test suite for the FrameProcessor class."""

    @pytest.fixture
    def mock_detector(self):
        """Fixture that returns a mocked detector."""
        detector = MagicMock()
        detector.detect.return_value = [
            Detection([10, 20, 30, 40], "test_class", 0.9)
        ]
        detector.get_name.return_value = "MockDetector"
        detector.get_info.return_value = {"model_type": "mock"}
        return detector

    @pytest.fixture
    def processor(self, mock_detector):
        """Fixture that returns a FrameProcessor with a mocked detector."""
        return FrameProcessor(mock_detector)

    def test_process_frame(self, processor):
        """Test that a frame can be processed correctly."""
        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process the frame
        frame_data = processor.process_frame(frame)
        
        # Check the result
        assert isinstance(frame_data, FrameData)
        assert frame_data.original_frame is not None
        assert frame_data.detections is not None
        assert len(frame_data.detections) == 1
        assert frame_data.annotated_frame is not None
        assert hasattr(frame_data, 'performance_stats')
        assert frame_data.detector is not None

    def test_get_stats(self, processor):
        """Test that stats can be retrieved."""
        # Process a frame to generate some stats
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor.process_frame(frame)
        
        # Get stats
        stats = processor.get_stats()
        
        # Check the stats
        assert isinstance(stats, dict)
        assert 'fps' in stats
        assert 'median_time' in stats
        assert 'max_time' in stats

    def test_real_detector_integration(self, test_image_path):
        """Test integration with a real detector."""
        # Create a real detector
        detector = create_detector(model_type="yolo")
        
        # Create a frame processor
        processor = FrameProcessor(detector)
        
        # Load a test image
        image = cv2.imread(test_image_path)
        assert image is not None
        
        # Process the frame
        frame_data = processor.process_frame(image)
        
        # Check the result has proper structure
        assert isinstance(frame_data, FrameData)
        assert frame_data.original_frame.shape == image.shape
        assert frame_data.annotated_frame.shape == image.shape
        assert frame_data.detector is detector
        
        # Check that it ran the detector
        assert len(frame_data.detections) > 0
        
        # Check that stats were collected
        assert 'fps' in frame_data.performance_stats
        assert 'median_time' in frame_data.performance_stats
        assert 'max_time' in frame_data.performance_stats


class TestFrameData:
    """Test suite for the FrameData class."""

    @pytest.fixture
    def frame_data(self):
        """Fixture that returns a FrameData instance."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [Detection([10, 20, 30, 40], "test_class", 0.9)]
        return FrameData(frame, detections)

    def test_initialization(self, frame_data):
        """Test that FrameData can be initialized correctly."""
        assert frame_data.original_frame is not None
        assert frame_data.detections is not None
        assert len(frame_data.detections) == 1
        assert frame_data.annotated_frame is None  # Not created until annotate_frame is called

    def test_annotate_frame(self, frame_data):
        """Test that frames can be annotated correctly."""
        # Annotate the frame
        frame_data.annotate_frame(stats_data={'fps': 30, 'median_time': 10, 'max_time': 20})
        
        # Check that the annotated frame was created
        assert frame_data.annotated_frame is not None
        assert frame_data.annotated_frame.shape == frame_data.original_frame.shape
        
        # Original frame should remain unchanged
        np.testing.assert_array_equal(
            frame_data.original_frame, 
            np.zeros((100, 100, 3), dtype=np.uint8)
        )