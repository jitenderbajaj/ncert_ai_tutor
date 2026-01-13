# FILE: backend/services/text_cleaner.py (NEW FILE)
"""
Text cleaning and preprocessing for NCERT PDF content

Removes noise before sending to LLM for summary generation:
- OCR artifacts and broken words
- Metadata, headers, footers
- Page numbers and formatting artifacts
- Excessive whitespace and special characters
- Navigation elements (Activity, Questions, etc.)
- Figure references without context
"""
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def clean_text_for_llm_summary(
    text: str,
    aggressive: bool = False,
    preserve_structure: bool = True
) -> str:
    """
    Clean extracted PDF text before sending to LLM for summary generation.
    
    Cleaning steps:
    1. Remove metadata and headers (Reprint, CHAP, page codes)
    2. Fix OCR errors and broken words
    3. Remove formatting artifacts
    4. Normalize whitespace
    5. Remove standalone page numbers
    6. Optionally remove navigation elements
    7. Fix common PDF extraction issues
    
    Args:
        text: Raw text extracted from PDF
        aggressive: If True, also remove Activities, Questions sections
        preserve_structure: If True, keep chapter/section headings
    
    Returns:
        Cleaned text suitable for LLM summary generation
    """
    logger.info(f"[TEXT CLEAN] Input: {len(text):,} chars, aggressive={aggressive}")
    
    original_text = text
    
    # Step 1: Remove common PDF metadata
    text = remove_pdf_metadata(text)
    
    # Step 2: Remove formatting artifacts
    text = remove_formatting_artifacts(text)
    
    # Step 3: Fix OCR errors
    text = fix_ocr_errors(text)
    
    # Step 4: Remove standalone page numbers
    text = remove_page_numbers(text)
    
    # Step 5: Clean whitespace
    text = normalize_whitespace(text)
    
    # Step 6: Remove navigation elements (optional)
    if aggressive:
        text = remove_navigation_elements(text)
    
    # Step 7: Remove figure references without context
    text = clean_figure_references(text)
    
    # Step 8: Fix sentence boundaries
    text = fix_sentence_boundaries(text)
    
    # Step 9: Remove very short lines (likely artifacts)
    if not preserve_structure:
        text = remove_short_lines(text, min_length=20)
    
    # Step 10: Final cleanup
    text = text.strip()
    
    reduction_pct = (1 - len(text) / len(original_text)) * 100
    logger.info(f"[TEXT CLEAN] Output: {len(text):,} chars ({reduction_pct:.1f}% reduction)")
    
    return text


def remove_pdf_metadata(text: str) -> str:
    """Remove PDF metadata, headers, footers"""
    
    patterns = [
        r'CHAP\s+\d+\.pmd',                    # CHAP 1.pmd
        r'\d{4}CH\d{2,4}',                     # 1064CH01, 1064CH02
        r'Reprint\s+\d{4}[-–]\d{2,4}',         # Reprint 2025-26
        r'NCERT.*?Reprint',                    # NCERT ... Reprint
        r'Copyright.*?\d{4}',                  # Copyright notices
        r'All rights reserved\.?',             # Rights reserved
        r'www\.ncert\.nic\.in',                # URLs
        r'ISBN\s*[:]\s*[\d-]+',                # ISBN numbers
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def remove_formatting_artifacts(text: str) -> str:
    """Remove common PDF extraction formatting artifacts"""
    
    # Common artifacts from PDF extraction
    artifacts = [
        'ished', 'shed', 'be re', 'republished', 'hotblished',
        'not to', 'no to', 'proc', 'ulphate', 'ma', 'da',
        'nc2.4.', 'quations', 'dicin', 'ublished'
    ]
    
    for artifact in artifacts:
        # Remove as standalone word
        text = re.sub(rf'\b{re.escape(artifact)}\b', '', text, flags=re.IGNORECASE)
    
    # Remove random single/double letters at line boundaries
    text = re.sub(r'\n[a-zA-Z]{1,2}\n', '\n', text)
    
    # Remove hyphenation artifacts (word- word)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    return text


def fix_ocr_errors(text: str) -> str:
    """Fix common OCR errors in NCERT PDFs"""
    
    # Common OCR corrections
    corrections = {
        'onsider': 'consider',
        'tawapannail': 'tawa/pan/nail',
        'copperII': 'copper(II)',
        'leadII': 'lead(II)',
        # 'O 2': 'O₂',
        # 'H 2': 'H₂',
        # 'CO 2': 'CO₂',
        # 'H2O 1': 'H₂O(l)',
        # 'H2O g': 'H₂O(g)',
        # 'H2O s': 'H₂O(s)',
        # 'HClaq': 'HCl(aq)',
        # 'NaOHaq': 'NaOH(aq)',
        # ' aq ': '(aq)',
        # ' g ': '(g)',
        # ' s ': '(s)',
        # ' l ': '(l)'
    }
    
    for error, correction in corrections.items():
        text = text.replace(error, correction)
    
    # Fix spaced chemical formulas (H 2 O → H₂O)
    # text = re.sub(r'([A-Z][a-z]?)\s+(\d+)', r'\1₂', text)
    
    return text


def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers"""
    
    # Page numbers at line boundaries
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    
    # Page numbers with surrounding whitespace
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: multiple spaces, tabs, newlines"""
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove spaces at line boundaries
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    
    return text


def remove_navigation_elements(text: str) -> str:
    """
    Remove navigation and structural elements (aggressive mode).
    
    Warning: This removes Activities, Questions, etc.
    Only use if you want pure conceptual content for summary.
    """
    
    # Remove activity blocks
    text = re.sub(
        r'Activity\s+\d+\.\d+.*?(?=Activity|\n\n[A-Z]|$)',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove question blocks at end of sections
    text = re.sub(
        r'QUESTIONS?.*?(?=\n\n[A-Z]|\d+\.\d+|$)',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove exercise sections
    text = re.sub(
        r'EXERCISES.*?$',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove "What you have learnt" sections
    text = re.sub(
        r'What you have learnt.*?(?=EXERCISES|$)',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return text


def clean_figure_references(text: str) -> str:
    """Clean figure references without context"""
    
    # Remove standalone figure references
    text = re.sub(r'Fig\.\s*\d+\.\d+[a-z]?(?!\s+shows|\s+illustrates)', 'a figure', text)
    text = re.sub(r'Figure\s+\d+\.\d+[a-z]?(?!\s+shows|\s+illustrates)', 'a figure', text)
    
    # Remove figure captions (usually multi-line)
    text = re.sub(
        r'Figure\s+\d+\.\d+\s+[A-Z].*?(?=\n\n)',
        '',
        text,
        flags=re.DOTALL
    )
    
    return text


def fix_sentence_boundaries(text: str) -> str:
    """Fix sentence boundary issues from PDF extraction"""
    
    # Add space after period if missing
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Fix question mark spacing
    text = re.sub(r'\?([A-Z])', r'? \1', text)
    
    # Fix exclamation mark spacing
    text = re.sub(r'!([A-Z])', r'! \1', text)
    
    return text


def remove_short_lines(text: str, min_length: int = 20) -> str:
    """Remove very short lines (likely artifacts)"""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        # Keep if long enough OR if it's a heading (starts with capital)
        if len(line_stripped) >= min_length or (line_stripped and line_stripped[0].isupper() and len(line_stripped) > 5):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_caution_and_warnings(text: str) -> str:
    """Remove CAUTION and WARNING blocks"""
    
    text = re.sub(
        r'CAUTION.*?(?=\n\n[A-Z]|Activity|\d+\.\d+)',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return text


def extract_core_content(text: str) -> str:
    """
    Extract only core educational content (most aggressive cleaning).
    
    Use this for summary generation when you want only:
    - Definitions
    - Explanations
    - Key concepts
    - Examples
    
    Removes:
    - Activities
    - Questions/Exercises
    - Do You Know sections
    - Cautions/Warnings
    """
    
    logger.info("[TEXT CLEAN] Extracting core content (aggressive mode)")
    
    # Remove all navigation and activity elements
    text = remove_navigation_elements(text)
    
    # Remove Do You Know blocks
    text = re.sub(
        r'Do You Know\?.*?(?=\n\n[A-Z]|\d+\.\d+|$)',
        '',
        text,
        flags=re.DOTALL
    )
    
    # Remove cautions
    text = remove_caution_and_warnings(text)
    
    # Remove group activities
    text = re.sub(
        r'Group Activity.*?$',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return text


def get_text_statistics(text: str) -> dict:
    """Get statistics about text for quality assessment"""
    
    lines = text.split('\n')
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        "total_chars": len(text),
        "total_lines": len(lines),
        "total_words": len(words),
        "total_sentences": len([s for s in sentences if s.strip()]),
        "avg_line_length": len(text) / len(lines) if lines else 0,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }


# Export public API
__all__ = [
    "clean_text_for_llm_summary",
    "extract_core_content",
    "get_text_statistics"
]
