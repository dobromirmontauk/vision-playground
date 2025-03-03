"""
YOLO object detection model implementation.
"""

import os
import contextlib
import time
from typing import List, Dict, Any, Optional
import numpy as np

from .base import ObjectDetector, Detection


class YOLODetector(ObjectDetector):
    """YOLO model implementation for object detection."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.25):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLO model silently."""
        # Silence the YOLO model initialization output
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: Image as numpy array (BGR format, as from OpenCV)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            self._initialize_model()
            
        # Run inference silently
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            results = self.model(frame)
            
        detections = []
        
        if len(results) > 0:
            # Extract boxes, classes, and confidence scores
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            class_names = results[0].names
            
            for i, box in enumerate(boxes):
                confidence = float(confs[i])
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                class_id = int(classes[i])
                class_name = class_names[class_id]
                
                detections.append(Detection([x1, y1, x2, y2], class_name, confidence))
        
        return detections
    
    def get_name(self) -> str:
        """Return the name of the detector."""
        return "YOLO"
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "model_type": "YOLO",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold
        }