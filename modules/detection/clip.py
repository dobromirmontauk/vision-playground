"""
CLIP-based object detection model implementation.
"""

import os
import contextlib
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import open_clip
from transformers import AutoProcessor, AutoModelForObjectDetection

from .base import ObjectDetector, Detection


class CLIPDetector(ObjectDetector):
    """
    CLIP-based detector implementation using CLIP for zero-shot object classification
    and a pretrained object detection model for bounding boxes.
    """
    
    def __init__(
        self, 
        clip_model_name: str = "ViT-B-32", 
        clip_pretrained: str = "laion2b_s34b_b79k",
        detector_model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.25,
        categories: List[str] = None
    ):
        """
        Initialize the CLIP-based detector.
        
        Args:
            clip_model_name: CLIP model architecture
            clip_pretrained: CLIP pretrained weights identifier
            detector_model_name: Object detection model for bounding boxes
            confidence_threshold: Minimum confidence threshold for detections
            categories: Optional list of categories to detect (if None, will use COCO categories)
        """
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.detector_model_name = detector_model_name
        self.confidence_threshold = confidence_threshold
        
        # Use provided categories or COCO categories as fallback
        self.categories = categories or [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
            "toothbrush"
        ]
        
        # Log the number of categories
        print(f"CLIP detector initialized with {len(self.categories)} categories")
        
        # Format categories for CLIP
        self.category_texts = [f"a photo of a {category}" for category in self.categories]
        
        # Models
        self.clip_model = None
        self.clip_preprocess = None
        self.detector_model = None
        self.detector_processor = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the CLIP and detector models."""
        # Initialize CLIP model - don't silence output for debugging 
        try:
            print("Initializing CLIP model...")
            # Check for available devices - prioritize MPS for Apple Silicon
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print(f"Using Apple Silicon GPU via MPS: {device}")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using NVIDIA GPU: {device}")
            else:
                device = torch.device("cpu")
                print(f"Using CPU: {device}")
            
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=device
            )
            self.clip_model.eval()  # Set to evaluation mode
            print("CLIP model initialized successfully")
            
            # Initialize object detector
            print("Initializing object detector model...")
            self.detector_processor = AutoProcessor.from_pretrained(self.detector_model_name)
            self.detector_model = AutoModelForObjectDetection.from_pretrained(self.detector_model_name).to(device)
            self.detector_model.eval()  # Set to evaluation mode
            print("Object detector model initialized successfully")
            
            # Store the device for later use
            self.device = device
            
            # Pre-compute text features for categories
            print("Computing text features for categories...")
            self.text_features = self._compute_text_features()
            print(f"Text features computed for {len(self.categories)} categories")
        except Exception as e:
            import traceback
            print(f"Error initializing models: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def _compute_text_features(self) -> torch.Tensor:
        """Compute text features for all categories using CLIP."""
        with torch.no_grad():
            text_tokens = open_clip.tokenize(self.category_texts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using CLIP for classification.
        
        Args:
            frame: Image as numpy array (BGR format, as from OpenCV)
            
        Returns:
            List of Detection objects
        """
        try:
            print("Starting detection...")
            if self.clip_model is None or self.detector_model is None:
                print("Models not initialized, initializing now...")
                self._initialize_models()
            
            print(f"Input frame shape: {frame.shape}")
            
            # Convert BGR to RGB for CLIP and PIL
            rgb_frame = frame[:, :, ::-1]
            pil_image = Image.fromarray(rgb_frame)
            print(f"Converted frame to PIL Image: {pil_image.size}")
            
            # Get bounding boxes from object detector
            print("Running object detection model...")
            with torch.no_grad():
                # Process image for detector
                inputs = self.detector_processor(images=pil_image, return_tensors="pt")
                # Move inputs to the same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                print(f"Processed inputs shape: {inputs['pixel_values'].shape}")
                
                outputs = self.detector_model(**inputs)
                print("Model inference complete")
                
                # Convert outputs to COCO API format
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                print(f"Target sizes: {target_sizes}")
                
                results = self.detector_processor.post_process_object_detection(
                    outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
                )[0]
                
                print(f"Detected {len(results['boxes'])} potential objects")
            
            detections = []
            
            # Process each detected box
            for i, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
                try:
                    confidence = float(score)
                    print(f"Processing detection {i+1}, initial confidence: {confidence:.2f}")
                    
                    # Skip low confidence detections
                    if confidence < self.confidence_threshold:
                        print(f"  Skipping: confidence {confidence:.2f} below threshold {self.confidence_threshold}")
                        continue
                    
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box.tolist())
                    print(f"  Box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                    
                    # Ensure valid box dimensions
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                        print(f"  Skipping: invalid box dimensions")
                        continue
                    
                    # Crop image to the detection box
                    crop = rgb_frame[y1:y2, x1:x2]
                    print(f"  Crop dimensions: {crop.shape}")
                    
                    # Skip if crop is too small
                    if crop.shape[0] < 10 or crop.shape[1] < 10:
                        print(f"  Skipping: crop too small")
                        continue
                    
                    # Convert crop to PIL and preprocess for CLIP
                    crop_pil = Image.fromarray(crop)
                    # Use a try-except block in case of any MPS-specific tensor issues
                    try:
                        crop_tensor = self.clip_preprocess(crop_pil).unsqueeze(0).to(self.device)
                        print(f"  Preprocessed tensor shape: {crop_tensor.shape}")
                    except Exception as e:
                        print(f"  Error processing tensor on {self.device}: {e}")
                        # Fall back to CPU if there's a device-specific error
                        crop_tensor = self.clip_preprocess(crop_pil).unsqueeze(0).to("cpu")
                        print(f"  Falling back to CPU tensor with shape: {crop_tensor.shape}")
                    
                    # Get CLIP image features
                    with torch.no_grad():
                        print("  Computing CLIP image features...")
                        try:
                            image_features = self.clip_model.encode_image(crop_tensor)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity with all categories
                            print("  Computing similarities with categories...")
                            similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                            
                            # Get the most likely class
                            top_prob, top_idx = similarities[0].topk(1)
                            
                            class_name = self.categories[top_idx.item()]
                            class_confidence = top_prob.item()
                            print(f"  CLIP classification: {class_name} with confidence {class_confidence:.2f}")
                            
                            # Use the CLIP confidence as the final confidence
                            final_confidence = class_confidence
                        except Exception as e:
                            print(f"  Error in CLIP processing: {e}")
                            # Default to the detector's classification
                            try:
                                # Use the detector's label as fallback
                                class_id = int(label.item())
                                class_name = self.detector_model.config.id2label.get(class_id, "unknown")
                                class_confidence = confidence
                                print(f"  Using detector's classification as fallback: {class_name}")
                            except Exception:
                                # If all else fails, use "unknown"
                                class_name = "unknown"
                                class_confidence = confidence
                                print(f"  Using 'unknown' as fallback classification")
                            final_confidence = class_confidence
                    
                    detections.append(Detection([x1, y1, x2, y2], class_name, final_confidence))
                    print(f"  Added detection: {class_name} at {[x1, y1, x2, y2]} with confidence {final_confidence:.2f}")
                except Exception as e:
                    print(f"  Error processing detection {i+1}: {e}")
                    continue
            
            print(f"Returning {len(detections)} final detections")
            return detections
        except Exception as e:
            import traceback
            print(f"Error in detect method: {e}")
            print(traceback.format_exc())
            return []
    
    def get_name(self) -> str:
        """Return the name of the detector."""
        return "CLIP-detector"
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about the detector."""
        return {
            "model_type": "CLIP-detector",
            "clip_model": f"{self.clip_model_name}-{self.clip_pretrained}",
            "detector_model": self.detector_model_name,
            "confidence_threshold": self.confidence_threshold,
            "categories": len(self.categories)
        }
        
    def get_device_info(self) -> str:
        """Return information about the inference device being used."""
        if hasattr(self, 'device'):
            # Return the device we're actually using
            return str(self.device)
        
        # Fallback to checking what's available
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except (ImportError, AttributeError):
            pass
            
        return "cpu"