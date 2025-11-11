"""
Text Extractor Module
Purpose: Extract text from PDFs that contain selectable text (non-scanned PDFs).
Uses pdfplumber as primary method, with PyPDF2 as fallback.

Logic:
1. Try to extract text from each page using pdfplumber
2. If pdfplumber fails, fallback to PyPDF2
3. Return a dictionary with page numbers as keys and extracted text as values
4. Track which pages have no text (likely image-based, need OCR)
"""

import pdfplumber
import PyPDF2
from typing import Dict, List
from src.utils.logger import get_logger

logger = get_logger()


class TextExtractor:
    """
    Extracts text from PDFs using pdfplumber (primary) and PyPDF2 (fallback).
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the text extractor.
        
        Args:
            pdf_path: Path to the PDF file to extract text from
        """
        self.pdf_path = pdf_path
        logger.info(f"Initialized TextExtractor for: {pdf_path}")
    
    def extract_with_pdfplumber(self) -> Dict[int, str]:
        """
        Extract text using pdfplumber (more accurate for most PDFs).
        
        Returns:
            Dictionary mapping page numbers (1-indexed) to extracted text
        """
        extracted_pages = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Extracting text from {total_pages} pages using pdfplumber")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text from the page
                        text = page.extract_text()
                        
                        if text:
                            extracted_pages[page_num] = text.strip()
                            logger.debug(f"Page {page_num}: Extracted {len(text)} characters")
                        else:
                            extracted_pages[page_num] = ""
                            logger.warning(f"Page {page_num}: No text found (possibly image-based)")
                    
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num} with pdfplumber: {e}")
                        extracted_pages[page_num] = ""
            
            logger.info(f"pdfplumber extraction complete: {len(extracted_pages)} pages processed")
            return extracted_pages
        
        except Exception as e:
            logger.error(f"pdfplumber failed to open PDF: {e}")
            return {}
    
    def extract_with_pypdf2(self) -> Dict[int, str]:
        """
        Extract text using PyPDF2 (fallback method).
        
        Returns:
            Dictionary mapping page numbers (1-indexed) to extracted text
        """
        extracted_pages = {}
        
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {total_pages} pages using PyPDF2")
                
                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text:
                            extracted_pages[page_num + 1] = text.strip()  # 1-indexed
                            logger.debug(f"Page {page_num + 1}: Extracted {len(text)} characters")
                        else:
                            extracted_pages[page_num + 1] = ""
                            logger.warning(f"Page {page_num + 1}: No text found")
                    
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num + 1} with PyPDF2: {e}")
                        extracted_pages[page_num + 1] = ""
            
            logger.info(f"PyPDF2 extraction complete: {len(extracted_pages)} pages processed")
            return extracted_pages
        
        except Exception as e:
            logger.error(f"PyPDF2 failed to open PDF: {e}")
            return {}
    
    def extract_text(self, use_fallback: bool = True) -> Dict[int, str]:
        """
        Extract text from PDF. Try pdfplumber first, fallback to PyPDF2 if needed.
        
        Args:
            use_fallback: Whether to use PyPDF2 as fallback if pdfplumber fails
        
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        logger.info(f"Starting text extraction from: {self.pdf_path}")
        
        # Try pdfplumber first (more reliable for most PDFs)
        extracted_pages = self.extract_with_pdfplumber()
        
        # If pdfplumber completely failed and fallback is enabled, try PyPDF2
        if not extracted_pages and use_fallback:
            logger.warning("pdfplumber extraction failed, trying PyPDF2 fallback")
            extracted_pages = self.extract_with_pypdf2()
        
        # Count pages with no text (will need OCR)
        empty_pages = [page_num for page_num, text in extracted_pages.items() if not text]
        
        if empty_pages:
            logger.warning(f"{len(empty_pages)} pages have no extractable text: {empty_pages}")
            logger.info("These pages may require OCR processing")
        
        logger.info(f"Text extraction complete: {len(extracted_pages)} total pages")
        return extracted_pages
    
    def get_pages_needing_ocr(
        self, 
        extracted_pages: Dict[int, str],
        check_hybrid: bool = True,
        hybrid_threshold: int = 100
    ) -> List[int]:
        """
        Identify which pages have no text OR suspiciously little text (hybrid pages).
        
        Args:
            extracted_pages: Dictionary of page numbers to extracted text
            check_hybrid: Whether to check for hybrid pages (text + images)
            hybrid_threshold: Character count below which a page is considered potentially hybrid
        
        Returns:
            List of page numbers that need OCR
        """
        ocr_pages = []
        hybrid_pages = []
        
        for page_num, text in extracted_pages.items():
            if not text:
                # Completely empty - definitely needs OCR
                ocr_pages.append(page_num)
            elif check_hybrid and len(text.strip()) < hybrid_threshold:
                # Has some text but very little - might be hybrid (text + images)
                hybrid_pages.append(page_num)
                ocr_pages.append(page_num)
        
        if ocr_pages:
            logger.info(f"Pages requiring OCR (empty): {[p for p in ocr_pages if p not in hybrid_pages]}")
        if hybrid_pages:
            logger.info(f"Pages requiring OCR (potential hybrid): {hybrid_pages}")
        if not ocr_pages:
            logger.info("All pages have sufficient extractable text, no OCR needed")
        
        return ocr_pages


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    
    # Setup logger for testing
    setup_logger()
    
    # Example usage
    pdf_path = "data/raw_pdfs/sample.pdf"
    
    extractor = TextExtractor(pdf_path)
    pages = extractor.extract_text()
    
    # Show results
    for page_num, text in pages.items():
        if text:
            print(f"\n--- Page {page_num} ---")
            print(text[:200] + "..." if len(text) > 200 else text)
        else:
            print(f"\n--- Page {page_num} (EMPTY - needs OCR) ---")
    
    # Check which pages need OCR
    ocr_needed = extractor.get_pages_needing_ocr(pages, check_hybrid=True, hybrid_threshold=100)
    print(f"\nPages needing OCR: {ocr_needed}")
