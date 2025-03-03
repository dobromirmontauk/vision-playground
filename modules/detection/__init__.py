"""
Detection module for object detection.
"""

from .base import Detection, ObjectDetector
from .yolo import YOLODetector


def create_detector(model_type: str = "yolo", **kwargs):
    """
    Factory function to create an object detector.
    
    Args:
        model_type: Type of detector to create (e.g., "yolo")
        **kwargs: Additional arguments to pass to the detector constructor
        
    Returns:
        An instance of ObjectDetector
        
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type.lower() == "yolo":
        return YOLODetector(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")