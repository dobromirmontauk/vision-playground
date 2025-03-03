#!/usr/bin/env python3
"""
Test script for the CLIP-based object detector.
This script creates a simple test that loads the detector and runs it on a test image.
"""

import os
import cv2
import time
import argparse
import numpy as np

from modules.detection import create_detector


def test_detector(model_type="clip", image_path=None, custom_categories=None):
    """Test the detector on a single image."""
    print(f"Testing {model_type} detector...")
    
    # Create detector with optional custom categories
    kwargs = {}
    if custom_categories:
        print(f"Using custom categories: {custom_categories}")
        kwargs['categories'] = custom_categories
    
    detector = create_detector(model_type=model_type, **kwargs)
    print(f"Detector info: {detector.get_info()}")
    
    # Load image - use webcam if no image path provided
    if image_path and os.path.exists(image_path):
        print(f"Loading image from {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return
    else:
        print("No valid image path provided, capturing from webcam...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame from webcam")
            cap.release()
            return
        cap.release()
        
        # Save the captured frame for reference
        cv2.imwrite("captured_frame.jpg", frame)
        print("Saved webcam frame to 'captured_frame.jpg'")
    
    # Measure detection time
    start_time = time.time()
    detections = detector.detect(frame)
    elapsed = time.time() - start_time
    
    print(f"Detection completed in {elapsed:.2f} seconds")
    print(f"Found {len(detections)} objects:")
    
    # Draw detections on the frame
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.box
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Print detection details
        print(f"  {i+1}. {detection.class_name} (confidence: {detection.confidence:.2f}) at {detection.box}")
    
    # Save the result
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test object detection models")
    parser.add_argument("--model", default="clip", choices=["yolo", "clip"], help="Model type to test")
    parser.add_argument("--image", help="Path to test image (optional, will use webcam if not provided)")
    parser.add_argument("--categories", nargs='+', help="Custom categories for detection (optional)")
    
    args = parser.parse_args()
    
    test_detector(
        model_type=args.model,
        image_path=args.image,
        custom_categories=args.categories
    )