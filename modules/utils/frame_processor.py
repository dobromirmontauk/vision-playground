"""
Frame processing utilities.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, NamedTuple
from ..detection.base import Detection
from .stats import PerformanceMonitor


class FrameData:
    """Class to hold frame data and metadata."""
    
    def __init__(self, 
                 original_frame: np.ndarray, 
                 detections: List[Detection] = None,
                 timestamp: float = None,
                 detector = None):
        """
        Initialize frame data.
        
        Args:
            original_frame: The original frame without annotations
            detections: List of Detection objects
            timestamp: Frame timestamp (defaults to current time)
            detector: Reference to the detector that processed this frame
        """
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.original_frame = original_frame
        self.detections = detections if detections is not None else []
        self.detector = detector
        
        # Create an annotated copy of the frame
        self.annotated_frame = None
    
    def annotate_frame(self, show_stats: bool = True, stats_data: Dict[str, Any] = None) -> np.ndarray:
        """
        Create and return a frame with visual annotations.
        
        Args:
            show_stats: Whether to show performance stats on the frame
            stats_data: Performance statistics to display
            
        Returns:
            Annotated frame
        """
        # Create a copy of the original frame
        self.annotated_frame = self.original_frame.copy()
        
        # Draw bounding boxes and labels
        for i, detection in enumerate(self.detections):
            x1, y1, x2, y2 = detection.box
            class_name = detection.class_name
            confidence = detection.confidence
            
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
            cv2.rectangle(self.annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(self.annotated_frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            
            # Choose text color (black or white) based on background brightness
            text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
            
            # Draw label text
            cv2.putText(self.annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Draw "Fix" button - make it bigger and more visible
            button_width = 50
            button_height = 30
            button_x = x2 - button_width
            button_y = y1
            cv2.rectangle(self.annotated_frame, (button_x, button_y), (x2, button_y + button_height), 
                         (0, 0, 255), -1)
            cv2.putText(self.annotated_frame, "FIX", (button_x + 10, button_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add detection ID overlay
            det_id = f"id:{i}"
            cv2.putText(self.annotated_frame, det_id, (x1 + 5, y2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add performance stats to frame
        if show_stats and stats_data:
            stats_text = f"FPS: {stats_data['fps']} | Median: {stats_data['median_time']} ms | Max: {stats_data['max_time']} ms"
            cv2.putText(self.annotated_frame, stats_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return self.annotated_frame


class FrameProcessor:
    """
    Process frames from a camera using an object detector.
    """
    
    def __init__(self, detector):
        """
        Initialize the frame processor.
        
        Args:
            detector: Object detector instance
        """
        self.detector = detector
        self.performance_monitor = PerformanceMonitor()
    
    def process_frame(self, frame: np.ndarray) -> FrameData:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            FrameData with the original frame, detections and annotations
        """
        # Start timing
        start_time = self.performance_monitor.start_processing()
        
        # Make a copy to ensure we don't modify the original frame
        frame_copy = frame.copy()
        
        # Run object detection
        detections = self.detector.detect(frame_copy)
        
        # End timing and update stats
        self.performance_monitor.end_processing(start_time)
        
        # Get the current performance stats
        stats = self.performance_monitor.get_stats()
        
        # Create frame data
        frame_data = FrameData(
            original_frame=frame_copy, 
            detections=detections,
            detector=self.detector
        )
        
        # Add performance stats to the frame data for retrieval by API
        frame_data.performance_stats = stats
        
        # Create annotated frame with stats
        frame_data.annotate_frame(show_stats=True, stats_data=stats)
        
        return frame_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics with device info."""
        stats = self.performance_monitor.get_stats()
        
        # Add device info to stats
        if self.detector:
            stats['device'] = self.detector.get_device_info()
        else:
            stats['device'] = "unknown"
            
        return stats