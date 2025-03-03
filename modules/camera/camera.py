"""
Camera implementations for capturing video frames.
"""

import time
import abc
import cv2
import numpy as np
from typing import Tuple, Optional


class Camera(abc.ABC):
    """Abstract base class for camera implementations."""
    
    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Return True if the camera is opened."""
        pass
    
    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        pass
    
    @abc.abstractmethod
    def release(self) -> None:
        """Release the camera resources."""
        pass


class OpenCVCamera(Camera):
    """Camera implementation using OpenCV VideoCapture."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the camera.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the camera."""
        self.camera = cv2.VideoCapture(self.camera_id)
        
        if self.camera.isOpened():
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
    
    def is_opened(self) -> bool:
        """Return True if the camera is opened."""
        return self.camera is not None and self.camera.isOpened()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened():
            return False, None
        
        return self.camera.read()
    
    def release(self) -> None:
        """Release the camera resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


class TestCamera(Camera):
    """Test camera that generates synthetic frames."""
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize the test camera.
        
        Args:
            width: Frame width
            height: Frame height
        """
        self.width = width
        self.height = height
        self.frame_count = 0
        self.opened = True
    
    def is_opened(self) -> bool:
        """Return True if the camera is opened."""
        return self.opened
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Generate a synthetic frame.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.opened:
            return False, None
        
        # Create a test frame with a moving rectangle to simulate motion
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw a gray background
        frame[:] = (100, 100, 100)
        
        # Draw a moving rectangle (simulating an object to detect)
        x = 100 + int(100 * np.sin(self.frame_count / 30))
        y = 200 + int(50 * np.cos(self.frame_count / 20))
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
        
        # Draw another rectangle
        x2 = 400 + int(50 * np.sin(self.frame_count / 25))
        y2 = 300 + int(30 * np.cos(self.frame_count / 15))
        cv2.rectangle(frame, (x2, y2), (x2 + 70, y2 + 70), (0, 0, 255), -1)
        
        # Add text explaining this is a test frame
        cv2.putText(frame, "TEST MODE - No camera access", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.frame_count += 1
        return True, frame
    
    def release(self) -> None:
        """Release the camera resources."""
        self.opened = False


def create_camera(use_test_camera: bool = False, max_attempts: int = 3, **kwargs) -> Camera:
    """
    Create and initialize a camera.
    
    Args:
        use_test_camera: Whether to use a test camera
        max_attempts: Maximum number of attempts to initialize the camera
        **kwargs: Additional arguments for the camera
        
    Returns:
        Initialized Camera instance
    """
    if use_test_camera:
        print("Using test camera with simulated objects")
        return TestCamera(**kwargs)
    
    # Try to initialize the real camera
    for attempt in range(max_attempts):
        camera = OpenCVCamera(**kwargs)
        
        if camera.is_opened():
            print(f"Camera initialized successfully on attempt {attempt+1}/{max_attempts}")
            return camera
        else:
            print(f"Camera initialization attempt {attempt+1}/{max_attempts} failed.")
            camera.release()
            time.sleep(1)
    
    # If we get here, camera initialization failed
    print("ERROR: Could not access the camera after multiple attempts!")
    print("On macOS, you might need to grant camera permissions to Terminal/iTerm2.")
    print("Go to System Preferences > Security & Privacy > Privacy > Camera")
    print("Using test camera as fallback.")
    
    return TestCamera(**kwargs)