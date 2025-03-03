"""
API utility functions.
"""

import os
import json
from datetime import datetime
import cv2
from PIL import Image
from typing import Dict, Any


def save_detection(frame, detection):
    """
    Save a detected object to disk.
    
    Args:
        frame: The original frame
        detection: Detection object as dictionary
        
    Returns:
        Filename of the saved image, or None if failed
    """
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Extract the bounding box and crop the image
        x1, y1, x2, y2 = detection['box']
        cropped_img = frame[y1:y2, x1:x2]
        
        # Convert to RGB for PIL
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_img_rgb)
        
        # Save the cropped image
        filename = f"{timestamp}_{detection['class']}_{detection['confidence']}.jpg"
        filepath = os.path.join('static/saved_boxes', filename)
        pil_img.save(filepath)
        
        # Save metadata
        metadata = {
            'original_class': detection['class'],
            'confidence': detection['confidence'],
            'timestamp': timestamp,
            'new_label': None,
            'filename': filename
        }
        
        metadata_path = os.path.join('static/saved_boxes', f"{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return filename
    except Exception as e:
        print(f"Error saving detection: {e}")
        return None