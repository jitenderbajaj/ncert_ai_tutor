# FILE: backend/multimodal/image_extract.py

"""
Page rendering with diagram detection and text box filtering
Filters out text-only boxes while keeping scientific diagrams
Version: 2.1.0 - Enhanced with text box detection
"""

from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path
import json
import io
import fitz
from PIL import Image
import numpy as np
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
RENDER_DPI = 200
MIN_REGION_WIDTH = 150
MIN_REGION_HEIGHT = 150
MIN_REGION_AREA = 30000


def extract_images_from_pdf(
    pdf_path: str,
    book_id: str,
    chapter_id: str,
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract diagrams by rendering pages and detecting visual diagram regions."""
    
    if output_dir is None:
        shard_dir = Path(settings.shards_dir) / f"{book_id}_{chapter_id}"
        output_dir = shard_dir / "images"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[IMAGE EXTRACT] Rendering pages to detect diagrams from {pdf_path}")
    logger.info(f"[IMAGE EXTRACT] Output: {output_dir}")
    
    doc = fitz.open(pdf_path)
    image_entries = []
    saved_images = set()
    
    stats = {
        "pages_processed": 0,
        "regions_detected": 0,
        "text_boxes_filtered": 0,
        "diagrams_saved": 0
    }
    
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            stats["pages_processed"] += 1
            
            # Render page at high resolution
            mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data))
            
            # Detect diagram regions
            regions = detect_diagram_regions(page_image, page_num)
            stats["regions_detected"] += len(regions)
            
            # Extract each region
            for region_idx, region_info in enumerate(regions):
                bbox, region_type, confidence = region_info
                
                # Filter text boxes
                if region_type == "text":
                    stats["text_boxes_filtered"] += 1
                    logger.info(f"[FILTER] Text box page {page_num+1}, region {region_idx+1} (conf: {confidence:.2f})")
                    continue
                
                # Crop region
                region_img = page_image.crop(bbox)
                
                # Save diagram
                stats["diagrams_saved"] += 1
                image_id = f"{book_id}_{chapter_id}_p{page_num+1}_fig{region_idx+1}"
                image_filename = f"{image_id}.png"
                image_path = output_dir / image_filename
                
                if image_filename not in saved_images:
                    region_img.save(image_path, "PNG")
                    saved_images.add(image_filename)
                    
                    width, height = region_img.size
                    logger.info(f"[IMAGE EXTRACT] ✓ DIAGRAM: {image_filename} ({width}x{height}, conf: {confidence:.2f})")
                
                # Create entry
                entry = {
                    "id": image_id,
                    "image_id": image_id,
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "page": page_num + 1,
                    "bbox": list(bbox),
                    "path": str(image_path),
                    "filename": image_filename,
                    "format": "png",
                    "size_bytes": image_path.stat().st_size if image_path.exists() else 0,
                    "extraction_method": "page_render",
                    "region_type": region_type,
                    "confidence": confidence,
                    "caption": None
                }
                image_entries.append(entry)
                
        except Exception as e:
            logger.error(f"[IMAGE EXTRACT] Error processing page {page_num+1}: {e}")
            continue
    
    doc.close()
    
    # Write manifest
    jsonl_path = output_dir / "images.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in image_entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"[IMAGE EXTRACT] ========================================")
    logger.info(f"[IMAGE EXTRACT] === EXTRACTION STATISTICS ===")
    logger.info(f"[IMAGE EXTRACT] Pages processed: {stats['pages_processed']}")
    logger.info(f"[IMAGE EXTRACT] Regions detected: {stats['regions_detected']}")
    logger.info(f"[IMAGE EXTRACT] Text boxes filtered: {stats['text_boxes_filtered']}")
    logger.info(f"[IMAGE EXTRACT] ✓✓✓ DIAGRAMS SAVED: {stats['diagrams_saved']} ✓✓✓")
    logger.info(f"[IMAGE EXTRACT] ========================================")
    
    return image_entries


def detect_diagram_regions(page_image: Image.Image, page_num: int) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Detect diagram regions in rendered page."""
    try:
        img_array = np.array(page_image)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array
        
        height, width = gray.shape
        
        # Find non-white regions
        non_white_mask = gray < 240
        
        # Find connected components
        regions = find_connected_regions(non_white_mask, min_size=MIN_REGION_AREA)
        
        # Filter and classify regions
        diagram_regions = []
        for region_bbox in regions:
            left, top, right, bottom = region_bbox
            region_width = right - left
            region_height = bottom - top
            region_area = region_width * region_height
            
            # Size filter
            if region_width < MIN_REGION_WIDTH or region_height < MIN_REGION_HEIGHT:
                continue
            
            if region_area < MIN_REGION_AREA:
                continue
            
            # Extract region for analysis
            region_mask = non_white_mask[top:bottom, left:right]
            region_img = gray[top:bottom, left:right]
            
            # Classify region
            region_type, confidence = classify_region(region_mask, region_img)
            
            # Add all regions (text will be filtered later)
            if (region_type == "diagram" and confidence > 0.5) or region_type == "text":
                # Add padding
                pad = 10
                padded_bbox = (
                    max(0, left - pad),
                    max(0, top - pad),
                    min(width, right + pad),
                    min(height, bottom + pad)
                )
                diagram_regions.append((padded_bbox, region_type, confidence))
        
        return diagram_regions
        
    except Exception as e:
        logger.error(f"[REGION DETECT] Error page {page_num+1}: {e}")
        return []


def find_connected_regions(mask: np.ndarray, min_size: int = 1000) -> List[Tuple[int, int, int, int]]:
    """Find connected regions (bounding boxes) in binary mask."""
    regions = []
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    
    def flood_fill(start_y, start_x):
        """Flood fill to find connected component."""
        stack = [(start_y, start_x)]
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        size = 0
        
        while stack:
            y, x = stack.pop()
            
            if y < 0 or y >= height or x < 0 or x >= width:
                continue
            if visited[y, x] or not mask[y, x]:
                continue
            
            visited[y, x] = True
            size += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
            # 4-connectivity
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        if size >= min_size:
            return (int(min_x), int(min_y), int(max_x), int(max_y))
        return None
    
    # Scan for regions
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            if mask[y, x] and not visited[y, x]:
                bbox = flood_fill(y, x)
                if bbox:
                    regions.append(bbox)
    
    return regions


def classify_region(region_mask: np.ndarray, region_img: np.ndarray) -> Tuple[str, float]:
    """
    Classify region as diagram or text box.
    Enhanced with text density detection.
    """
    try:
        height, width = region_mask.shape
        
        # Basic features
        fill_ratio = region_mask.sum() / (height * width)
        aspect_ratio = max(width, height) / min(width, height)
        
        # Edge analysis
        edges_h = np.abs(np.diff(region_img.astype(float), axis=0))
        edges_v = np.abs(np.diff(region_img.astype(float), axis=1))
        edge_density = (edges_h.sum() + edges_v.sum()) / (height * width)
        
        # Horizontal edge dominance (text has strong horizontal patterns)
        total_edges = edges_h.sum() + edges_v.sum()
        if total_edges > 0:
            h_edge_ratio = edges_h.sum() / total_edges
        else:
            h_edge_ratio = 0.5
        
        # Detect regular horizontal spacing (text lines)
        row_sums = region_mask.sum(axis=1)
        if len(row_sums) > 20:
            row_variance = np.std(row_sums)
            row_mean = np.mean(row_sums)
            if row_mean > 0:
                row_cv = row_variance / row_mean
            else:
                row_cv = 0
        else:
            row_cv = 0
        
        # TEXT BOX DETECTION (STRICT)
        is_text_box = (
            fill_ratio < 0.25 and  # Sparse (text has white space)
            h_edge_ratio > 0.60 and  # Strong horizontal edges
            edge_density > 0.08 and  # Has structure
            row_cv > 0.3  # Regular line spacing
        )
        
        if is_text_box:
            logger.debug(f"[TEXT BOX] fill: {fill_ratio:.2f}, h_ratio: {h_edge_ratio:.2f}, edge: {edge_density:.2f}, row_cv: {row_cv:.2f}")
            return ("text", 0.2)
        
        # Traditional text-like (very wide/tall)
        is_text_like = (
            fill_ratio < 0.3 and
            aspect_ratio > 3.5 and
            edge_density > 0.1
        )
        
        if is_text_like:
            return ("text", 0.3)
        
        # DIAGRAM features
        is_diagram = (
            0.15 < fill_ratio < 0.75 and
            aspect_ratio < 4.0 and
            edge_density > 0.05 and
            h_edge_ratio < 0.70  # NOT predominantly horizontal
        )
        
        if is_diagram:
            # Confidence calculation
            confidence = min(1.0, fill_ratio + (edge_density / 0.2))
            
            # Penalize high horizontal edge ratio (text-like)
            if h_edge_ratio > 0.65:
                confidence *= 0.5
            
            return ("diagram", confidence)
        
        return ("other", 0.2)
        
    except Exception as e:
        logger.error(f"[CLASSIFY ERROR] {e}")
        return ("unknown", 0.0)


__all__ = ["extract_images_from_pdf"]



