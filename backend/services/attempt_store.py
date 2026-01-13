# FILE: backend/services/attempt_store.py
"""
Attempt store with idempotent submit
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import csv
import io

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AttemptStore:
    """Store for student attempts"""
    
    def __init__(self):
        self.attempts_dir = Path(settings.attempts_dir)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
    
    def submit(
        self,
        attempt_id: str,
        user_id: str,
        question_id: str,
        book_id: str,
        chapter_id: str,
        response: str,
        correctness: Optional[float] = None,
        bloom_level: Optional[str] = None,
        hots_level: Optional[str] = None,
        evaluation: Optional[Dict[str, Any]] = None
    ):
        """Submit attempt (idempotent)"""
        attempts_file = self.attempts_dir / f"{book_id}_{chapter_id}.jsonl"
        
        # Check if attempt already exists
        existing = self._get_attempt(attempts_file, attempt_id)
        if existing:
            logger.debug(f"Attempt {attempt_id} already exists (idempotent)")
            return
        
        # Create attempt record
        attempt = {
            "attempt_id": attempt_id,
            "user_id": user_id,
            "question_id": question_id,
            "book_id": book_id,
            "chapter_id": chapter_id,
            "response": response,
            "correctness": correctness,
            "bloom_level": bloom_level,
            "hots_level": hots_level,
            "evaluation": evaluation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Append to file
        with open(attempts_file, 'a') as f:
            f.write(json.dumps(attempt) + '\n')
        
        logger.info(f"Submitted attempt: {attempt_id}")
    
    def _get_attempt(self, attempts_file: Path, attempt_id: str) -> Optional[Dict[str, Any]]:
        """Get existing attempt"""
        if not attempts_file.exists():
            return None
        
        with open(attempts_file, 'r') as f:
            for line in f:
                attempt = json.loads(line)
                if attempt["attempt_id"] == attempt_id:
                    return attempt
        
        return None
    
    def export(
        self,
        book_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        user_id: Optional[str] = None,
        format: str = "csv"
    ) -> Any:
        """Export attempts"""
        # Collect attempts
        attempts = []
        
        if book_id and chapter_id:
            attempts_file = self.attempts_dir / f"{book_id}_{chapter_id}.jsonl"
            if attempts_file.exists():
                with open(attempts_file, 'r') as f:
                    for line in f:
                        attempt = json.loads(line)
                        if user_id is None or attempt["user_id"] == user_id:
                            attempts.append(attempt)
        else:
            # Export all
            for attempts_file in self.attempts_dir.glob("*.jsonl"):
                with open(attempts_file, 'r') as f:
                    for line in f:
                        attempt = json.loads(line)
                        if user_id is None or attempt["user_id"] == user_id:
                            attempts.append(attempt)
        
        # Format output
        if format == "csv":
            return self._export_csv(attempts)
        else:
            return attempts
    
    def _export_csv(self, attempts: List[Dict[str, Any]]) -> str:
        """Export attempts as CSV"""
        if not attempts:
            return ""
        
        output = io.StringIO()
        fieldnames = ["attempt_id", "user_id", "question_id", "book_id", "chapter_id",
                     "response", "correctness", "bloom_level", "hots_level", "timestamp"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for attempt in attempts:
            row = {k: attempt.get(k, "") for k in fieldnames}
            writer.writerow(row)
        
        return output.getvalue()
