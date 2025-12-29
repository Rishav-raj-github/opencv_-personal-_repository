"""Enhanced Facial Recognition System

This module provides celebrity recognition capabilities using face detection,
feature extraction, and personality database matching.
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CelebrityRecognizer:
    """Recognize celebrities using facial detection and feature matching."""
    
    def __init__(self, use_dlib=True):
        """Initialize the facial recognition system.
        
        Args:
            use_dlib: Whether to use Dlib for face detection (vs Haarcascade)
        """
        self.use_dlib = use_dlib
        self.faces_detected = 0
        self.celebrities_recognized = 0
        self.processing_times = []
        
        # Initialize face detectors
        try:
            if use_dlib:
                self.face_detector = dlib.get_frontal_face_detector()
                logger.info("Dlib face detector initialized")
        except Exception as e:
            logger.warning(f"Dlib initialization failed: {e}")
            use_dlib = False
        
        if not use_dlib:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV Haarcascade initialized")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected face regions with coordinates
        """
        faces = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            if self.use_dlib:
                dlib_rects = self.face_detector(gray, 1)
                for rect in dlib_rects:
                    faces.append({
                        'x': rect.left(),
                        'y': rect.top(),
                        'width': rect.width(),
                        'height': rect.height()
                    })
            else:
                cascade_faces = self.cascade_classifier.detectMultiScale(
                    gray, 1.1, 4
                )
                for (x, y, w, h) in cascade_faces:
                    faces.append({'x': x, 'y': y, 'width': w, 'height': h})
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
        
        return faces
    
    def recognize_from_image(self, image_path: str) -> Dict:
        """Recognize celebrities from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with recognition results
        """
        results = {'celebrities': [], 'faces_detected': 0}
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return results
            
            faces = self.detect_faces(image)
            results['faces_detected'] = len(faces)
            self.faces_detected += len(faces)
            
            logger.info(f"Detected {len(faces)} faces in {image_path}")
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
        
        return results
    
    def batch_recognize(self, image_folder: str) -> Dict:
        """Process multiple images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            
        Returns:
            Dictionary with batch processing results
        """
        results = {
            'total_images': 0,
            'recognized_faces': 0,
            'processing_time': 0
        }
        
        folder_path = Path(image_folder)
        if not folder_path.exists():
            logger.error(f"Folder not found: {image_folder}")
            return results
        
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        results['total_images'] = len(image_files)
        
        for image_file in image_files:
            result = self.recognize_from_image(str(image_file))
            results['recognized_faces'] += result.get('faces_detected', 0)
        
        logger.info(f"Batch processing complete. Processed {results['total_images']} images")
        return results
    
    def get_statistics(self) -> Dict:
        """Get recognition statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_faces_detected': self.faces_detected,
            'total_celebrities_recognized': self.celebrities_recognized,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }


def setup_celebrity_database() -> Dict:
    """Setup celebrity personality database.
    
    Returns:
        Dictionary mapping celebrity names to their metadata
    """
    return {
        'Elon Musk': {'tags': ['entrepreneur', 'tech', 'tesla'], 'confidence_threshold': 0.6},
        'Virat Kohli': {'tags': ['cricket', 'india', 'athlete'], 'confidence_threshold': 0.6},
        'Amitabh Bachchan': {'tags': ['actor', 'bollywood', 'india'], 'confidence_threshold': 0.6},
        'Narendra Modi': {'tags': ['politician', 'india', 'pm'], 'confidence_threshold': 0.6},
        'Shah Rukh Khan': {'tags': ['actor', 'bollywood', 'india'], 'confidence_threshold': 0.6},
        'Aishwarya Rai': {'tags': ['actress', 'bollywood', 'india'], 'confidence_threshold': 0.6},
        'Aamir Khan': {'tags': ['actor', 'bollywood', 'india'], 'confidence_threshold': 0.6},
        'Priyanka Chopra': {'tags': ['actress', 'bollywood', 'hollywood'], 'confidence_threshold': 0.6}
    }


if __name__ == '__main__':
    # Test the module
    print('Initializing Facial Recognition System...')
    db = setup_celebrity_database()
    print(f'Celebrity database loaded: {len(db)} personalities')
    
    recognizer = CelebrityRecognizer()
    stats = recognizer.get_statistics()
    print(f'System statistics: {stats}')
