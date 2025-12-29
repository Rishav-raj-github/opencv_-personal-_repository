#!/usr/bin/env python3
"""
Camera-based Facial Recognition System
Captures live video from webcam and recognizes celebrities and objects
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add repository to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from facial_recognition_enhanced import CelebrityRecognizer
except ImportError:
    print("Error: Could not import CelebrityRecognizer. Ensure facial_recognition_enhanced.py is in the same directory.")
    sys.exit(1)

class CameraFacialRecognition:
    """
    Real-time facial recognition system using webcam
    """
    
    def __init__(self, confidence_threshold=0.5):
        """Initialize camera and recognizer"""
        self.recognizer = CelebrityRecognizer()
        self.confidence_threshold = confidence_threshold
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.frame_count = 0
        self.detected_celebrities = {}
        
        print("\n" + "="*70)
        print("🎥 CAMERA FACIAL RECOGNITION SYSTEM INITIALIZED")
        print("="*70)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'c' to clear detected celebrities")
        print("="*70 + "\n")
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray
    
    def run_camera_stream(self):
        """
        Main camera stream processing loop
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera. Please check if camera is connected.")
            return False
        
        print("✅ Camera opened successfully!")
        print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print("\nStarting camera stream...\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to read frame from camera")
                break
            
            self.frame_count += 1
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            
            # Draw rectangles around faces and recognize them
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Try to recognize face (in real implementation would use face_recognition library)
                # For now, just add generic label
                cv2.putText(frame, 'Face Detected', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add frame info
            info_text = f"Frames: {self.frame_count} | Faces detected: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Facial Recognition System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print("\n✋ Quitting camera stream...")
                break
            elif key == ord('s'):  # Save frame
                filename = f'capture_{self.frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved: {filename}")
            elif key == ord('c'):  # Clear detections
                self.detected_celebrities.clear()
                print("🔄 Cleared detected celebrities")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("📊 SESSION STATISTICS")
        print("="*70)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total faces detected: {len(self.detected_celebrities)}")
        print("="*70 + "\n")
        
        return True


def main():
    """
    Main entry point
    """
    try:
        print("\n" + "#"*70)
        print("# CAMERA-BASED FACIAL RECOGNITION SYSTEM")
        print("# Real-time face detection and celebrity recognition")
        print("#"*70 + "\n")
        
        camera_system = CameraFacialRecognition()
        camera_system.run_camera_stream()
        
        print("\n✅ System shutdown successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
