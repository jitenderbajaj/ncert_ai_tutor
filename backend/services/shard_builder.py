# FILE: backend/services/shard_builder.py
"""
Shard builder utilities
"""
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Chunk text with overlap and stable passage IDs
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        seed: Random seed for determinism
    
    Returns:
        List of chunk dicts
    """
    # Split into sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "passage_id": f"passage_{chunk_id}",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "image_anchors": []  # Would be populated during image binding
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-(overlap // 50):] if overlap > 0 else []
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "passage_id": f"passage_{chunk_id}",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "image_anchors": []
        })
    
    logger.debug(f"Chunked text into {len(chunks)} chunks")
    
    return chunks


def build_faiss_index(embeddings, factory_string: str = "IndexFlatL2"):
    """Build FAISS index from embeddings"""
    import faiss
    import numpy as np
    
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    if factory_string == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.index_factory(dimension, factory_string)
    
    index.add(embeddings)
    
    return index
