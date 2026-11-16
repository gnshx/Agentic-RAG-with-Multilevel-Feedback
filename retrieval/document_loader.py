import os
import hashlib
from typing import List, Dict, Any
from core.state import DocumentChunk

class DocumentLoader:
    """
    Loads raw text files, parses into structured chunks with consistent IDs,
    hash fingerprints, and source metadata.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "manual") -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        text = text.strip()
        if not text:
            return chunks

        # Split into paragraphs or approximate character windows
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = ""
        chunk_idx = 0

        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}".strip() if current_chunk else p
            else:
                if current_chunk:
                    cid = hashlib.md5(f"{source}_{chunk_idx}_{current_chunk[:50]}".encode()).hexdigest()[:12]
                    chunks.append({
                        "id": cid,
                        "content": current_chunk,
                        "source": source,
                        "score": 0.0,
                        "dynamic_weight": 0.0,
                        "metadata": {"chunk_index": chunk_idx, "char_len": len(current_chunk)}
                    })
                    chunk_idx += 1
                current_chunk = p

        if current_chunk:
            cid = hashlib.md5(f"{source}_{chunk_idx}_{current_chunk[:50]}".encode()).hexdigest()[:12]
            chunks.append({
                "id": cid,
                "content": current_chunk,
                "source": source,
                "score": 0.0,
                "dynamic_weight": 0.0,
                "metadata": {"chunk_index": chunk_idx, "char_len": len(current_chunk)}
            })

        return chunks

    def load_directory(self, dir_path: str) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        if not os.path.exists(dir_path):
            return all_chunks

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".txt", ".md", ".json")):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        chunks = self.chunk_text(content, source=file)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        return all_chunks

