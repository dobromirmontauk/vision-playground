"""
Performance monitoring and statistics.
"""

import time
import collections
import numpy as np
from typing import Dict, Any


class PerformanceMonitor:
    """Monitor performance metrics like FPS and processing times."""
    
    def __init__(self, max_samples: int = 100, fps_window: int = 50):
        """
        Initialize the performance monitor.
        
        Args:
            max_samples: Maximum number of processing time samples to keep
            fps_window: Number of frame timestamps to use for FPS calculation
        """
        self.processing_times = collections.deque(maxlen=max_samples)
        self.fps_counter = collections.deque(maxlen=fps_window)
        self.stats = {
            'fps': 0,
            'median_time': 0,
            'max_time': 0
        }
    
    def start_processing(self) -> float:
        """
        Start timing for processing.
        
        Returns:
            Current timestamp
        """
        return time.time()
    
    def end_processing(self, start_time: float) -> float:
        """
        End timing for processing and update stats.
        
        Args:
            start_time: Start timestamp from start_processing()
            
        Returns:
            Processing time in milliseconds
        """
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        # Update FPS counter
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        self._update_stats()
        
        return processing_time
    
    def _update_stats(self) -> None:
        """Update all statistics."""
        # Update FPS
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                self.stats['fps'] = round(len(self.fps_counter) / time_diff, 1)
        
        # Update processing time stats
        if self.processing_times:
            self.stats['median_time'] = round(np.median(self.processing_times), 1)
            self.stats['max_time'] = round(max(self.processing_times), 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current performance statistics."""
        return self.stats.copy()