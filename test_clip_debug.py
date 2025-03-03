#!/usr/bin/env python3
"""
Debug test for the CLIP detector.
This script tests the CLIP detector on static images to help debug any issues.
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import traceback

from modules.detection import create_detector, CLIPDetector


def test_clip_detector(image_path, categories=None, verbose=True):
    """
    Test the CLIP detector on a single image.
    
    Args:
        image_path: Path to the test image
        categories: Optional list of custom categories
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (success, detections, error_message)
    """
    try:
        if not os.path.exists(image_path):
            return False, [], f"Image file not found: {image_path}"
        
        if verbose:
            print(f"Testing CLIP detector on image: {image_path}")
        
        # Create detector
        kwargs = {}
        if categories:
            kwargs['categories'] = categories
            if verbose:
                print(f"Using custom categories: {categories}")
        
        # Initialize the detector
        if verbose:
            print("Initializing CLIP detector...")
        
        start_time = time.time()
        detector = create_detector(model_type="clip", **kwargs)
        init_time = time.time() - start_time
        
        if verbose:
            print(f"Detector initialized in {init_time:.2f} seconds")
            print(f"Detector info: {detector.get_info()}")
        
        # Load the image
        if verbose:
            print("Loading image...")
        
        image = cv2.imread(image_path)
        if image is None:
            return False, [], f"Failed to load image: {image_path}"
        
        image_height, image_width = image.shape[:2]
        if verbose:
            print(f"Image dimensions: {image_width}x{image_height}")
        
        # Perform detection
        if verbose:
            print("Running detection...")
        
        start_time = time.time()
        detections = detector.detect(image)
        detection_time = time.time() - start_time
        
        if verbose:
            print(f"Detection completed in {detection_time:.2f} seconds")
            print(f"Found {len(detections)} objects:")
            
            for i, detection in enumerate(detections):
                print(f"  {i+1}. {detection.class_name} (confidence: {detection.confidence:.2f}) at {detection.box}")
        
        # Create output image with bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save the output image
        output_path = "clip_detection_result.jpg"
        cv2.imwrite(output_path, image)
        if verbose:
            print(f"Saved annotated image to {output_path}")
        
        return True, detections, f"Detection successful, found {len(detections)} objects"
    
    except Exception as e:
        error_message = f"Error testing CLIP detector: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_message)
        return False, [], error_message


def debug_clip_implementation(verbose=True):
    """Debug the core CLIP detector implementation."""
    try:
        if verbose:
            print("Debugging CLIP detector implementation...")
        
        # Test detector creation
        detector = CLIPDetector()
        
        if verbose:
            print("Successfully created detector instance")
            print(f"Model info: {detector.get_info()}")
        
        # Test initialization of internal models
        if verbose:
            print("Testing model initialization...")
        
        if detector.clip_model is None:
            detector._initialize_models()
        
        if detector.clip_model is not None and detector.detector_model is not None:
            if verbose:
                print("Successfully initialized models")
            return True, "Models initialized successfully"
        else:
            if verbose:
                print("Failed to initialize models")
            return False, "Failed to initialize models"
    
    except Exception as e:
        error_message = f"Error debugging CLIP implementation: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_message)
        return False, error_message


def test_multiple_images(directory="static/saved_boxes", max_images=5, categories=None):
    """
    Test the CLIP detector on multiple images in a directory.
    
    Args:
        directory: Directory containing images
        max_images: Maximum number of images to test
        categories: Custom categories to use
    """
    print(f"Testing CLIP detector on images in {directory}")
    
    # Find all JPG images in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:max_images]  # Limit the number of images
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    print(f"Found {len(image_files)} images to test")
    
    results = []
    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        print(f"\nTesting image {i+1}/{len(image_files)}: {filename}")
        
        success, detections, message = test_clip_detector(image_path, categories, verbose=True)
        results.append((filename, success, len(detections), message))
    
    # Print summary
    print("\n--- Summary ---")
    for filename, success, num_detections, message in results:
        status = "SUCCESS" if success and num_detections > 0 else "FAILED"
        print(f"{filename}: {status} - {num_detections} detections")
    
    # Check if any test was successful
    any_success = any(success and num_detections > 0 for _, success, num_detections, _ in results)
    if any_success:
        print("\nAt least one image was successfully processed!")
    else:
        print("\nAll tests failed. Check the error messages above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug test for CLIP detector")
    parser.add_argument("--image", help="Path to a specific test image")
    parser.add_argument("--dir", default="static/saved_boxes", help="Directory containing test images")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to test")
    parser.add_argument("--categories", nargs='+', help="Custom categories to detect")
    parser.add_argument("--debug-implementation", action="store_true", help="Debug the CLIP implementation")
    
    args = parser.parse_args()
    
    if args.debug_implementation:
        success, message = debug_clip_implementation()
        sys.exit(0 if success else 1)
    
    if args.image:
        # Test a specific image
        success, detections, message = test_clip_detector(args.image, args.categories)
        sys.exit(0 if success and len(detections) > 0 else 1)
    else:
        # Test multiple images
        test_multiple_images(args.dir, args.max_images, args.categories)