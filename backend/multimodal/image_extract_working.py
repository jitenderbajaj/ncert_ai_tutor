# FILE: backend/multimodal/image_extract.py
"""
Enhanced image extraction from PDFs with fallback strategies

Handles CMYK, JPEG2000, inline images, and corrupted extractions
Compatible with bind_images_to_chunks() in ingestion_dual.py

Version: 0.11.4
"""
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import json
import io

import fitz  # PyMuPDF
from PIL import Image

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def extract_images_from_pdf(
    pdf_path: str,
    book_id: str,
    chapter_id: str,
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract images from PDF with enhanced strategies for problematic formats.
    
    Extraction strategies (in order):
    1. Standard PyMuPDF extraction (extract_image)
    2. Pixmap rendering (for inline/corrupted images)
    3. Color space conversion (CMYK → RGB)
    
    Args:
        pdf_path: Path to PDF file
        book_id: Book identifier
        chapter_id: Chapter identifier
        output_dir: Optional output directory
    
    Returns:
        List of image manifest entries
    """
    # Default output directory
    if output_dir is None:
        shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
        output_dir = shard_dir / "images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[IMAGE EXTRACT] Extracting from {pdf_path}")
    logger.info(f"[IMAGE EXTRACT] Output: {output_dir}")
    
    doc = fitz.open(pdf_path)
    image_entries = []
    saved_images = set()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Strategy 1: Extract XObject images (standard method)
        image_list = page.get_images(full=True)
        
        logger.debug(f"[IMAGE EXTRACT] Page {page_num+1}: {len(image_list)} images found")
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                
                # Try standard extraction first
                extraction_result = extract_image_with_fallback(
                    doc=doc,
                    page=page,
                    xref=xref,
                    page_num=page_num,
                    img_index=img_index
                )
                
                if extraction_result is None:
                    logger.warning(f"[IMAGE EXTRACT] Failed on page {page_num+1}, img {img_index+1}")
                    continue
                
                image_bytes, image_ext, is_fallback = extraction_result
                
                # Check if image is valid (not all black/corrupted)
                if not is_valid_image(image_bytes):
                    logger.warning(f"[IMAGE EXTRACT] Invalid/black image on page {page_num+1}, img {img_index+1}")
                    # Try pixmap fallback
                    image_bytes, image_ext, is_fallback = extract_via_pixmap(
                        page=page,
                        xref=xref,
                        page_num=page_num,
                        img_index=img_index
                    )
                    
                    if image_bytes is None or not is_valid_image(image_bytes):
                        logger.error(f"[IMAGE EXTRACT] All strategies failed for page {page_num+1}, img {img_index+1}")
                        continue
                
                # Generate filenames
                image_base_id = f"{book_id}_{chapter_id}_p{page_num+1}_img{img_index+1}"
                image_filename = f"{image_base_id}.{image_ext}"
                image_path = output_dir / image_filename
                
                # Save image
                if image_filename not in saved_images:
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    saved_images.add(image_filename)
                    logger.debug(f"[IMAGE EXTRACT] Saved: {image_filename} ({len(image_bytes):,} bytes)")
                
                # Get bounding boxes
                bbox_list = page.get_image_rects(xref)
                
                if not bbox_list:
                    # No bbox info, use default
                    bbox_list = [fitz.Rect(0, 0, 100, 100)]
                
                # Create manifest entries
                for instance_index, bbox_rect in enumerate(bbox_list):
                    image_id = f"{image_base_id}_inst{instance_index+1}" if len(bbox_list) > 1 else image_base_id
                    
                    entry = {
                        "id": image_id,
                        "image_id": image_id,
                        "book_id": book_id,
                        "chapter_id": chapter_id,
                        "page": page_num + 1,
                        "bbox": [bbox_rect.x0, bbox_rect.y0, bbox_rect.x1, bbox_rect.y1],
                        "path": str(image_path),
                        "filename": image_filename,
                        "format": image_ext,
                        "size_bytes": len(image_bytes),
                        "extraction_method": "fallback" if is_fallback else "standard",
                        "caption": None
                    }
                    
                    image_entries.append(entry)
            
            except Exception as e:
                logger.error(f"[IMAGE EXTRACT] Error on page {page_num+1}, img {img_index+1}: {e}")
                continue
    
    doc.close()
    
    # Write manifest
    jsonl_path = output_dir / "images.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in image_entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"[IMAGE EXTRACT] ✓ Extracted {len(image_entries)} instances from {len(saved_images)} unique images")
    
    return image_entries


def extract_image_with_fallback(
    doc: fitz.Document,
    page: fitz.Page,
    xref: int,
    page_num: int,
    img_index: int
) -> Optional[tuple]:
    """
    Extract image with fallback strategies.
    
    Returns:
        (image_bytes, image_ext, is_fallback) or None
    """
    try:
        # Strategy 1: Standard extraction
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # Check color space and convert if needed
        colorspace = base_image.get("colorspace", 0)
        
        if colorspace == 3:  # CMYK
            logger.debug(f"[IMAGE EXTRACT] CMYK image detected, converting to RGB")
            image_bytes = convert_cmyk_to_rgb(image_bytes, image_ext)
            image_ext = "png"  # Save as PNG after conversion
        
        return (image_bytes, image_ext, False)
    
    except Exception as e:
        logger.warning(f"[IMAGE EXTRACT] Standard extraction failed: {e}")
        return None


def extract_via_pixmap(
    page: fitz.Page,
    xref: int,
    page_num: int,
    img_index: int
) -> tuple:
    """
    Extract image by rendering page region as pixmap (fallback method).
    
    This works for inline images and corrupted XObjects.
    
    Returns:
        (image_bytes, image_ext, is_fallback)
    """
    try:
        logger.debug(f"[IMAGE EXTRACT] Using pixmap fallback for page {page_num+1}, img {img_index+1}")
        
        # Get image bounding box
        bbox_list = page.get_image_rects(xref)
        
        if not bbox_list:
            logger.warning("[IMAGE EXTRACT] No bbox for pixmap rendering")
            return (None, None, True)
        
        bbox = bbox_list[0]  # Use first occurrence
        
        # Render page region as pixmap (high quality)
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=bbox)
        
        # Convert to PNG bytes
        image_bytes = pix.tobytes("png")
        
        logger.debug(f"[IMAGE EXTRACT] Pixmap rendered: {len(image_bytes):,} bytes")
        
        return (image_bytes, "png", True)
    
    except Exception as e:
        logger.error(f"[IMAGE EXTRACT] Pixmap fallback failed: {e}")
        return (None, None, True)


def convert_cmyk_to_rgb(image_bytes: bytes, image_ext: str) -> bytes:
    """
    Convert CMYK image to RGB using PIL.
    
    Args:
        image_bytes: CMYK image bytes
        image_ext: Image format (jpg, png, etc.)
    
    Returns:
        RGB image bytes as PNG
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert CMYK to RGB
        if img.mode == 'CMYK':
            logger.debug("[IMAGE EXTRACT] Converting CMYK → RGB")
            img = img.convert('RGB')
        
        # Save as PNG
        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()
    
    except Exception as e:
        logger.error(f"[IMAGE EXTRACT] CMYK conversion failed: {e}")
        return image_bytes  # Return original if conversion fails


def is_valid_image(image_bytes: bytes) -> bool:
    """
    Check if image bytes represent a valid, non-black image.
    
    Detects:
    - Corrupted images
    - All-black images
    - Empty images
    
    Args:
        image_bytes: Image bytes to validate
    
    Returns:
        True if valid, False if corrupted/black
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Check if image is too small (likely corrupted)
        if img.width < 10 or img.height < 10:
            logger.debug("[IMAGE EXTRACT] Image too small (< 10x10 px)")
            return False
        
        # Convert to RGB for analysis
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Sample pixels (faster than checking all)
        import numpy as np
        
        # Resize to 100x100 for quick check
        img_small = img.resize((100, 100))
        pixels = np.array(img_small)
        
        # Check if image is mostly black
        mean_brightness = pixels.mean()
        
        if mean_brightness < 5:  # Almost black (0-255 scale)
            logger.debug(f"[IMAGE EXTRACT] Image is mostly black (brightness: {mean_brightness:.1f})")
            return False
        
        # Check variance (all same color = corrupted)
        variance = pixels.std()
        
        if variance < 1:  # No variation
            logger.debug(f"[IMAGE EXTRACT] Image has no variation (variance: {variance:.1f})")
            return False
        
        return True
    
    except Exception as e:
        logger.warning(f"[IMAGE EXTRACT] Validation failed: {e}")
        return True  # Assume valid if we can't check


# Export public API
__all__ = [
    "extract_images_from_pdf",
    "extract_image_with_fallback",
    "extract_via_pixmap",
    "convert_cmyk_to_rgb",
    "is_valid_image"
]
