# OCR for Document Images

## Project Description
This project implements Optical Character Recognition (OCR) for extracting text from document images. The system processes scanned documents, photos, and digital images to convert printed or handwritten text into machine-readable format. It includes preprocessing, text detection, and recognition capabilities.

## Features
- Text extraction from various document types
- Support for multiple languages
- Handwriting recognition
- Table and layout detection
- PDF document processing
- Batch processing capability
- Output in multiple formats (TXT, JSON, XML)
- Confidence scoring for extracted text

## Requirements
- Python 3.8+
- OpenCV
- Tesseract OCR or EasyOCR
- pytesseract
- Pillow (PIL)
- NumPy
- pdf2image (for PDF processing)

## Usage
The main implementation will be in `document_ocr.py`

## Processing Pipeline
1. **Document Loading**: Load images or PDFs
2. **Preprocessing**: Deskew, denoise, binarization
3. **Layout Analysis**: Detect text regions, tables, images
4. **Text Detection**: Locate text blocks and lines
5. **OCR**: Extract text from detected regions
6. **Post-processing**: Spell check, formatting, validation

## Supported Document Types
- Scanned documents
- Business cards
- Receipts and invoices
- Forms and surveys
- Handwritten notes
- Book pages and articles

## Preprocessing Techniques
- Grayscale conversion
- Noise reduction (Gaussian blur, median filter)
- Binarization (Otsu, adaptive threshold)
- Deskewing and rotation correction
- Border removal
- Contrast enhancement

## Expected Output
- Extracted text with position coordinates
- Structured data (JSON format)
- Searchable PDFs
- Word and character confidence scores
- Layout preservation
