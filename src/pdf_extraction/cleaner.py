"""
Text Cleaner Module
Purpose: Merge and clean extracted text from PDFs (both regular text and OCR).
Combines text from all pages, adds metadata, and prepares text for chunking.

Logic:
1. Merge text from text extraction and OCR extraction
2. Clean whitespace, remove extra blank lines
3. Optionally remove page numbers and headers/footers
4. Skip completely blank pages
5. Create structured JSON output with page-level metadata
6. Save as JSON for RAG pipeline
7. Return structured data ready for chunking
"""

import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.utils.logger import get_logger
from src.utils.file_utils import ensure_directory_exists, save_text_file, save_json

logger = get_logger()


class TextCleaner:
    """
    Cleans and merges extracted text from PDFs.
    Combines regular text extraction and OCR results into a single cleaned document.
    """
    
    def __init__(
        self, 
        clean_whitespace: bool = True,
        remove_page_numbers: bool = True,
        min_page_length: int = 20
    ):
        """
        Initialize the text cleaner.
        
        Args:
            clean_whitespace: Whether to normalize whitespace and remove extra blank lines
            remove_page_numbers: Whether to attempt removing standalone page numbers
            min_page_length: Minimum character count to consider a page non-empty
        """
        self.clean_whitespace = clean_whitespace
        self.remove_page_numbers = remove_page_numbers
        self.min_page_length = min_page_length
        
        logger.info("Initialized TextCleaner")
        logger.info(f"Settings: clean_whitespace={clean_whitespace}, remove_page_numbers={remove_page_numbers}")
    
    def merge_text_and_ocr(
        self, 
        text_pages: Dict[int, str], 
        ocr_pages: Dict[int, str] = None,
        merge_hybrid: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Merge regular text extraction with OCR results.
        
        For empty pages: OCR text replaces empty text
        For hybrid pages: OCR text is APPENDED to existing text (merge both)
        
        Args:
            text_pages: Dictionary of page numbers to extracted text
            ocr_pages: Dictionary of page numbers to OCR text (optional)
            merge_hybrid: If True, append OCR to existing text for hybrid pages
        
        Returns:
            Dictionary with page numbers as keys and page data (text + metadata) as values
        """
        logger.info("Merging text extraction and OCR results")
        
        merged = {}
        
        # Process all pages from text extraction
        for page_num, text in text_pages.items():
            merged[page_num] = {
                'text': text.strip(),
                'extraction_method': 'text',
                'has_ocr': False
            }
        
        # Merge OCR results
        if ocr_pages:
            for page_num, ocr_text in ocr_pages.items():
                if page_num in merged:
                    existing_text = merged[page_num]['text']
                    
                    if not existing_text:
                        # Empty page - use OCR text
                        merged[page_num]['text'] = ocr_text.strip()
                        merged[page_num]['extraction_method'] = 'ocr'
                        merged[page_num]['has_ocr'] = True
                        logger.debug(f"Page {page_num}: Using OCR text ({len(ocr_text)} chars)")
                    elif merge_hybrid and ocr_text.strip():
                        # Hybrid page - merge both texts
                        merged[page_num]['text'] = f"{existing_text}\n\n{ocr_text.strip()}"
                        merged[page_num]['extraction_method'] = 'text+ocr'
                        merged[page_num]['has_ocr'] = True
                        logger.debug(f"Page {page_num}: Merged text + OCR ({len(existing_text)} + {len(ocr_text)} chars)")
                else:
                    # Page not in original extraction - add from OCR
                    merged[page_num] = {
                        'text': ocr_text.strip(),
                        'extraction_method': 'ocr',
                        'has_ocr': True
                    }
                    logger.debug(f"Page {page_num}: Added from OCR")
        
        logger.info(f"Merged {len(merged)} pages total")
        return merged
    
    def clean_page_text(self, text: str) -> str:
        """
        Clean text from a single page.
        
        Args:
            text: Raw text from a page
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        if self.clean_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r' +', ' ', text)
            
            # Replace multiple newlines with double newline (paragraph break)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            # Remove trailing/leading whitespace from each line
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
        
        # Remove standalone page numbers (common pattern: digit(s) alone on a line)
        if self.remove_page_numbers:
            # Match lines that are just numbers (page numbers)
            text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
            
            # Match common page number patterns like "Page 5" or "- 5 -"
            text = re.sub(r'^(Page\s+\d+|[-–—]\s*\d+\s*[-–—])$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove any resulting multiple blank lines again
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def is_page_empty(self, text: str) -> bool:
        """
        Check if a page is essentially empty (too short to be meaningful).
        
        Args:
            text: Page text
        
        Returns:
            True if page should be considered empty
        """
        if not text:
            return True
        
        # Remove all whitespace and check length
        text_without_spaces = re.sub(r'\s+', '', text)
        
        return len(text_without_spaces) < self.min_page_length
    
    def create_structured_output(
        self, 
        pages: Dict[int, Dict[str, Any]], 
        pdf_name: str
    ) -> Dict[str, Any]:
        """
        Create structured JSON output with document and page-level metadata.
        
        Args:
            pages: Dictionary mapping page numbers to page data (text + metadata)
            pdf_name: Original PDF filename (without extension)
        
        Returns:
            Structured dictionary ready for JSON serialization
        """
        logger.info(f"Creating structured output for {len(pages)} pages")
        
        processed_pages = []
        skipped_pages = []
        
        # Sort pages by page number
        sorted_pages = sorted(pages.items(), key=lambda x: x[0])
        
        for page_num, page_data in sorted_pages:
            # Clean the page text
            cleaned_text = self.clean_page_text(page_data['text'])
            
            # Skip empty pages
            if self.is_page_empty(cleaned_text):
                skipped_pages.append(page_num)
                logger.debug(f"Page {page_num}: Skipped (empty or too short)")
                continue
            
            # Create page entry with metadata
            page_entry = {
                'page_number': page_num,
                'text': cleaned_text,
                'char_count': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'extraction_method': page_data['extraction_method'],
                'has_ocr': page_data['has_ocr']
            }
            
            processed_pages.append(page_entry)
            logger.debug(f"Page {page_num}: Added ({len(cleaned_text)} chars, {page_entry['word_count']} words)")
        
        # Create document structure
        document = {
            'metadata': {
                'pdf_name': pdf_name,
                'total_pages': len(pages),
                'processed_pages': len(processed_pages),
                'skipped_pages': len(skipped_pages),
                'skipped_page_numbers': skipped_pages,
                'extraction_date': datetime.now().isoformat(),
                'cleaning_settings': {
                    'clean_whitespace': self.clean_whitespace,
                    'remove_page_numbers': self.remove_page_numbers,
                    'min_page_length': self.min_page_length
                }
            },
            'pages': processed_pages
        }
        
        # Log summary
        if skipped_pages:
            logger.info(f"Skipped {len(skipped_pages)} empty pages: {skipped_pages}")
        
        total_chars = sum(p['char_count'] for p in processed_pages)
        total_words = sum(p['word_count'] for p in processed_pages)
        logger.info(f"Structured output: {len(processed_pages)} pages, {total_chars} chars, {total_words} words")
        
        return document
    
    def merge_pages_with_markers(self, pages: Dict[int, Dict[str, Any]]) -> str:
        """
        Merge all pages into a single text with [PAGE X] markers.
        (Legacy method - kept for backward compatibility with plain text output)
        
        Args:
            pages: Dictionary mapping page numbers to page data
        
        Returns:
            Single string with all pages merged and marked
        """
        logger.info(f"Merging {len(pages)} pages with [PAGE X] markers (legacy text format)")
        
        merged_text_parts = []
        skipped_pages = []
        
        # Sort pages by page number
        sorted_pages = sorted(pages.items(), key=lambda x: x[0])
        
        for page_num, page_data in sorted_pages:
            # Clean the page text
            cleaned_text = self.clean_page_text(page_data['text'])
            
            # Skip empty pages
            if self.is_page_empty(cleaned_text):
                skipped_pages.append(page_num)
                logger.debug(f"Page {page_num}: Skipped (empty or too short)")
                continue
            
            # Add page marker and text
            page_marker = f"[PAGE {page_num}]"
            merged_text_parts.append(f"{page_marker}\n{cleaned_text}")
            logger.debug(f"Page {page_num}: Added ({len(cleaned_text)} chars)")
        
        # Join all pages with double newline
        merged_text = "\n\n".join(merged_text_parts)
        
        # Log summary
        if skipped_pages:
            logger.info(f"Skipped {len(skipped_pages)} empty pages: {skipped_pages}")
        
        logger.info(f"Final merged text: {len(merged_text)} characters")
        return merged_text
    
    def clean_and_merge(
        self, 
        text_pages: Dict[int, str], 
        ocr_pages: Optional[Dict[int, str]] = None,
        merge_hybrid: bool = True,
        output_format: str = 'json'
    ) -> Any:
        """
        Main method: merge text and OCR, clean, and create structured output.
        
        Args:
            text_pages: Dictionary of page numbers to extracted text
            ocr_pages: Optional dictionary of page numbers to OCR text
            merge_hybrid: If True, merge OCR with existing text for hybrid pages
            output_format: 'json' for structured output (default) or 'text' for legacy format
        
        Returns:
            Structured dictionary (if json) or plain text with [PAGE X] markers (if text)
        """
        logger.info("Starting text cleaning and merging process")
        
        # Step 1: Merge text extraction and OCR with metadata
        merged_pages = self.merge_text_and_ocr(text_pages, ocr_pages, merge_hybrid)
        
        if output_format == 'json':
            # Return structured output (recommended for RAG)
            logger.info("Creating JSON structured output")
            return merged_pages  # Will be processed in create_structured_output
        else:
            # Return plain text with markers (legacy)
            logger.info("Creating plain text output (legacy)")
            final_text = self.merge_pages_with_markers(merged_pages)
            return final_text
    
    def save_cleaned_text(
        self, 
        cleaned_data: Any, 
        output_dir: str, 
        pdf_name: str,
        output_format: str = 'json'
    ) -> str:
        """
        Save cleaned data to a file (JSON or plain text).
        
        Args:
            cleaned_data: The cleaned data (dict for JSON, str for text)
            output_dir: Directory to save the file
            pdf_name: Original PDF filename (without extension)
            output_format: 'json' or 'text'
        
        Returns:
            Path to the saved file
        """
        ensure_directory_exists(output_dir)
        
        if output_format == 'json':
            # Create structured document if not already done
            if isinstance(cleaned_data, dict) and 'pages' not in cleaned_data:
                # cleaned_data is the merged_pages dict, need to structure it
                document = self.create_structured_output(cleaned_data, pdf_name)
            else:
                document = cleaned_data
            
            output_filename = f"{pdf_name}_extracted.json"
            output_path = os.path.join(output_dir, output_filename)
            save_json(document, output_path)
        else:
            # Plain text format
            output_filename = f"{pdf_name}_cleaned.txt"
            output_path = os.path.join(output_dir, output_filename)
            save_text_file(cleaned_data, output_path)
        
        logger.info(f"Cleaned data saved to: {output_path}")
        return output_path
    
    def get_cleaning_stats(self, original_pages: Dict[int, str], cleaned_data: Any) -> Dict[str, any]:
        """
        Get statistics about the cleaning process.
        
        Args:
            original_pages: Original pages before cleaning
            cleaned_data: Final cleaned data (dict for JSON, str for text)
        
        Returns:
            Dictionary with cleaning statistics
        """
        total_pages = len(original_pages)
        original_chars = sum(len(text) for text in original_pages.values())
        
        if isinstance(cleaned_data, dict):
            # JSON format
            if 'pages' in cleaned_data:
                kept_pages = len(cleaned_data['pages'])
                cleaned_chars = sum(p['char_count'] for p in cleaned_data['pages'])
            elif 'metadata' in cleaned_data:
                # Already structured
                kept_pages = cleaned_data['metadata']['processed_pages']
                cleaned_chars = sum(p['char_count'] for p in cleaned_data['pages'])
            else:
                # Merged pages dict
                kept_pages = len([p for p in cleaned_data.values() if not self.is_page_empty(p['text'])])
                cleaned_chars = sum(len(p['text']) for p in cleaned_data.values() if not self.is_page_empty(p['text']))
        else:
            # Plain text format
            kept_pages = len(re.findall(r'\[PAGE \d+\]', cleaned_data))
            cleaned_chars = len(cleaned_data)
        
        stats = {
            'total_pages': total_pages,
            'kept_pages': kept_pages,
            'skipped_pages': total_pages - kept_pages,
            'original_chars': original_chars,
            'cleaned_chars': cleaned_chars,
            'compression_ratio': round(cleaned_chars / original_chars, 2) if original_chars > 0 else 0
        }
        
        logger.info(f"Cleaning stats: {stats}")
        return stats


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    
    # Setup logger for testing
    setup_logger()
    
    # Example: simulate extracted pages
    text_pages = {
        1: "Introduction\n\nThis is the first page with some content.\n\n1",
        2: "Chapter 1: Background\n\nMore detailed text here.\n\n2",
        3: "",  # Completely empty page (needs OCR)
        4: "Fig 1:",  # Hybrid page - has a little text but mostly an image (needs OCR too)
        5: "Conclusion\n\nFinal thoughts.\n\n5"
    }
    
    # Example: simulate OCR results for pages 3 and 4
    ocr_pages = {
        3: "This page was scanned and required OCR to extract text.",
        4: "This diagram shows the architecture of the system with three main components."
    }
    
    # Create cleaner
    cleaner = TextCleaner(
        clean_whitespace=True,
        remove_page_numbers=True,
        min_page_length=20
    )
    
    # Clean and merge (JSON format - recommended for RAG)
    merged_pages = cleaner.clean_and_merge(text_pages, ocr_pages, merge_hybrid=True, output_format='json')
    
    # Create structured document
    document = cleaner.create_structured_output(merged_pages, pdf_name="sample_document")
    
    # Display result
    print("\n" + "="*50)
    print("STRUCTURED JSON OUTPUT:")
    print("="*50)
    print(f"Metadata: {document['metadata']}")
    print(f"\nTotal pages: {len(document['pages'])}")
    
    for page in document['pages'][:2]:  # Show first 2 pages
        print(f"\nPage {page['page_number']}:")
        print(f"  - Chars: {page['char_count']}, Words: {page['word_count']}")
        print(f"  - Method: {page['extraction_method']}")
        print(f"  - Text preview: {page['text'][:100]}...")
    
    # Get statistics
    stats = cleaner.get_cleaning_stats(text_pages, document)
    print("\n" + "="*50)
    print("CLEANING STATISTICS:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save as JSON
    output_path = cleaner.save_cleaned_text(
        cleaned_data=document,
        output_dir="data/extracted_texts",
        pdf_name="sample_document",
        output_format='json'
    )
    print(f"\n✅ Saved to: {output_path}")
