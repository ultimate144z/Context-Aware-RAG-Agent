"""
OCR Extractor Module
Purpose: Extract text from image-based/scanned PDFs using Optical Character Recognition (OCR).
Uses pdf2image to convert PDF pages to images, then pytesseract for text recognition.

Logic:
1. Convert specified PDF pages to images using pdf2image
2. Apply pytesseract OCR to each image to extract text
3. Save temporary images for debugging (optional)
4. Clean up temporary files after extraction
5. Return dictionary with page numbers and OCR-extracted text

Dependencies:
- Tesseract OCR must be installed on the system
- Poppler utils must be installed (for pdf2image)
"""

import os
import pytesseract
from pdf2image import convert_from_path
from typing import Dict, List, Optional
from PIL import Image
from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory_exists

logger = get_logger()


class OCRExtractor:
    """
    Extracts text from image-based PDFs using OCR (Optical Character Recognition).
    """
    
    def __init__(
        self, 
        pdf_path: str, 
        temp_image_dir: str = "data/temp_images",
        ocr_lang: str = "eng",
        tesseract_path: Optional[str] = None
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            pdf_path: Path to the PDF file
            temp_image_dir: Directory to save temporary images during OCR
            ocr_lang: Language for OCR ('eng' for English, 'urd' for Urdu, etc.)
            tesseract_path: Custom path to tesseract executable (if not in PATH)
        """
        self.pdf_path = pdf_path
        self.temp_image_dir = temp_image_dir
        self.ocr_lang = ocr_lang
        
        # Set tesseract path if provided (useful for Windows)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Ensure temp directory exists
        ensure_directory_exists(self.temp_image_dir)
        
        logger.info(f"Initialized OCRExtractor for: {pdf_path}")
        logger.info(f"OCR Language: {ocr_lang}")
    
    def convert_page_to_image(self, page_number: int, dpi: int = 300) -> Optional[Image.Image]:
        """
        Convert a specific PDF page to an image.
        
        Args:
            page_number: Page number to convert (1-indexed)
            dpi: Resolution for image conversion (higher = better quality but slower)
        
        Returns:
            PIL Image object or None if conversion fails
        """
        try:
            logger.debug(f"Converting page {page_number} to image at {dpi} DPI")
            
            # Convert specific page (first_page and last_page are 1-indexed)
            images = convert_from_path(
                self.pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number
            )
            
            if images:
                logger.debug(f"Successfully converted page {page_number} to image")
                return images[0]
            else:
                logger.error(f"No image generated for page {page_number}")
                return None
        
        except Exception as e:
            logger.error(f"Error converting page {page_number} to image: {e}")
            return None
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from an image using pytesseract OCR.
        
        Args:
            image: PIL Image object
        
        Returns:
            Extracted text as string
        """
        try:
            # Perform OCR on the image
            text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            logger.debug(f"OCR extracted {len(text)} characters")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return ""
    
    def ocr_page(
        self, 
        page_number: int, 
        dpi: int = 300, 
        save_image: bool = False
    ) -> str:
        """
        Perform OCR on a specific page.
        
        Args:
            page_number: Page number to OCR (1-indexed)
            dpi: Image resolution for OCR
            save_image: Whether to save the temporary image for debugging
        
        Returns:
            Extracted text from the page
        """
        logger.info(f"Starting OCR for page {page_number}")
        
        # Convert page to image
        image = self.convert_page_to_image(page_number, dpi)
        
        if image is None:
            logger.warning(f"Could not convert page {page_number} to image")
            return ""
        
        # Optionally save image for debugging
        if save_image:
            image_path = os.path.join(
                self.temp_image_dir, 
                f"page_{page_number}.png"
            )
            image.save(image_path)
            logger.debug(f"Saved temporary image: {image_path}")
        
        # Extract text using OCR
        text = self.extract_text_from_image(image)
        
        if text:
            logger.info(f"OCR successful for page {page_number}: {len(text)} characters extracted")
        else:
            logger.warning(f"OCR found no text on page {page_number}")
        
        return text
    
    def ocr_multiple_pages(
        self, 
        page_numbers: List[int], 
        dpi: int = 300,
        save_images: bool = False
    ) -> Dict[int, str]:
        """
        Perform OCR on multiple pages.
        
        Args:
            page_numbers: List of page numbers to OCR (1-indexed)
            dpi: Image resolution for OCR
            save_images: Whether to save temporary images
        
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        logger.info(f"Starting OCR for {len(page_numbers)} pages")
        
        ocr_results = {}
        
        for page_num in page_numbers:
            text = self.ocr_page(page_num, dpi, save_images)
            ocr_results[page_num] = text
        
        # Summary
        successful_pages = sum(1 for text in ocr_results.values() if text)
        logger.info(f"OCR complete: {successful_pages}/{len(page_numbers)} pages had extractable text")
        
        return ocr_results
    
    def cleanup_temp_images(self) -> None:
        """
        Delete all temporary images created during OCR.
        """
        try:
            if os.path.exists(self.temp_image_dir):
                for file in os.listdir(self.temp_image_dir):
                    file_path = os.path.join(self.temp_image_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleaned up temporary images from: {self.temp_image_dir}")
        
        except Exception as e:
            logger.error(f"Error cleaning up temp images: {e}")
    
    @staticmethod
    def is_tesseract_installed() -> bool:
        """
        Check if Tesseract OCR is installed and accessible.
        
        Returns:
            True if tesseract is available, False otherwise
        """
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version detected: {version}")
            return True
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            logger.error("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            return False


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    
    # Setup logger for testing
    setup_logger()
    
    # Check if Tesseract is installed
    if not OCRExtractor.is_tesseract_installed():
        print("Error: Tesseract OCR is not installed!")
        exit(1)
    
    # Example usage
    pdf_path = "data/raw_pdfs/scanned_document.pdf"
    
    # Create OCR extractor
    ocr = OCRExtractor(
        pdf_path=pdf_path,
        temp_image_dir="data/temp_images",
        ocr_lang="eng"
    )
    
    # OCR specific pages (e.g., pages 1, 2, 3)
    pages_to_ocr = [1, 2, 3]
    results = ocr.ocr_multiple_pages(pages_to_ocr, save_images=True)
    
    # Display results
    for page_num, text in results.items():
        print(f"\n--- Page {page_num} (OCR) ---")
        print(text[:300] + "..." if len(text) > 300 else text)
    
    # Clean up temporary images
    # ocr.cleanup_temp_images()
