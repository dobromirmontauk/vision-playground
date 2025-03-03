"""
Abstract base class for object detection models.
"""

import abc
from typing import List, Dict, Any, Tuple
import numpy as np


class Detection:
    """Represents a single object detection result."""
    
    def __init__(self, box: List[int], class_name: str, confidence: float):
        """
        Initialize a detection result.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            class_name: Class name of the detected object
            confidence: Confidence score (0-1)
        """
        self.box = box
        self.class_name = class_name
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'box': self.box,
            'class': self.class_name,
            'confidence': round(self.confidence, 2)
        }


class ObjectDetector(abc.ABC):
    """Abstract base class for object detection models."""
    
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Image as numpy array (BGR format, as from OpenCV)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """Return the name of the detector."""
        pass
    
    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        pass