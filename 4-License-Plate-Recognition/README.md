# License Plate Recognition

## Project Description
This project implements an Automatic License Plate Recognition (ALPR) system using computer vision and OCR techniques. The system detects vehicle license plates in images or video streams and extracts the alphanumeric characters for identification purposes.

## Features
- License plate detection in various lighting conditions
- Character segmentation and recognition
- Multi-country license plate format support
- Real-time processing capability
- Integration with OCR engines (Tesseract, EasyOCR)
- Vehicle detection and tracking
- Database logging of recognized plates

## Requirements
- Python 3.8+
- OpenCV
- Tesseract OCR or EasyOCR
- NumPy
- imutils
- scikit-image

## Usage
The main implementation will be in `license_plate_recognition.py`

## Pipeline Components
1. **Vehicle Detection**: Detect vehicles in the frame
2. **Plate Localization**: Locate license plate region
3. **Plate Enhancement**: Preprocess plate image (resize, denoise, threshold)
4. **Character Segmentation**: Separate individual characters
5. **OCR**: Extract text from plate
6. **Post-processing**: Validate and format results

## Dataset
- OpenALPR dataset
- Custom collected license plate images
- Synthetic license plate generation

## Techniques Used
- Morphological operations
- Edge detection (Canny, Sobel)
- Contour analysis
- Perspective transformation
- Template matching
- Deep learning (YOLO for detection)

## Expected Output
- Detected license plate regions
- Extracted text characters
- Confidence scores
- Processing time metrics
