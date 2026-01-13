# FILE: backend/services/assessment_store.py
"""
Assessment store
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AssessmentStore:
    """Store for assessments"""
    
    def __init__(self):
        self.data_dir = Path(settings.data_dir) / "assessments"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_assessments(
        self,
        book_id: str,
        chapter_id: str
    ) -> List[Dict[str, Any]]:
        """Get assessments for chapter"""
        assessment_file = self.data_dir / f"{book_id}_{chapter_id}.jsonl"
        
        if not assessment_file.exists():
            return []
        
        assessments = []
        with open(assessment_file, 'r') as f:
            for line in f:
                assessments.append(json.loads(line))
        
        return assessments
    
    def add_assessment(
        self,
        book_id: str,
        chapter_id: str,
        assessment: Dict[str, Any]
    ):
        """Add assessment"""
        assessment_file = self.data_dir / f"{book_id}_{chapter_id}.jsonl"
        
        with open(assessment_file, 'a') as f:
            f.write(json.dumps(assessment) + '\n')
