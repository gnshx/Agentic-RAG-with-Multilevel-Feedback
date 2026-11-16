import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
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


def ingest_document_to_stores(
    corpus: List[Dict[str, Any]],
    retriever: Any,
    vector_store: Any,
    source_name: str,
    content: str,
    title: Optional[str] = None,
    on_conflict: str = "rename"
) -> Dict[str, Any]:
    """
    Unified, synchronized document ingestion for both HybridRetriever and LocalVectorStore.
    Supports:
      - Raw text input (source_name="user_input")
      - Text files (.txt)
      - Markdown files (.md)
      - Structured JSON files (.json) - array of dicts, single dict, or array of strings.

    Collision & Deduplication Guarantees:
      1. Never treats matching IDs alone as proof of duplicate content.
      2. Compares normalized content and source before skipping.
      3. If an explicit ID already exists with different content:
         - When on_conflict="rename" (default): generates a deterministic alternative ID (e.g. f"{cid}_{hash[:8]}")
           to safely index both documents without data loss.
         - When on_conflict="error": raises ValueError with a clear collision explanation.
      4. Preserves identical IDs and metadata across HybridRetriever and LocalVectorStore.
      5. Preserves source + content-hash deduplication (same content + different IDs).
      6. Synchronizes stores if a document exists in one store but is missing from the other.
      7. Repeated uploads remain strictly idempotent.

    Returns:
      {
          "added_count": int,
          "skipped_count": int,
          "total_corpus": int,
          "total_vstore": int,
          "chunk_ids": List[str]
      }
    """
    if not content or not content.strip():
        return {
            "added_count": 0,
            "skipped_count": 0,
            "total_corpus": len(corpus),
            "total_vstore": len(vector_store.documents) if vector_store and hasattr(vector_store, "documents") else 0,
            "chunk_ids": []
        }

    candidates: List[Dict[str, Any]] = []

    # 1. Parse candidates by format
    if source_name.lower().endswith(".json"):
        loaded_json = json.loads(content)
        if isinstance(loaded_json, list):
            items = loaded_json
        elif isinstance(loaded_json, dict):
            if "corpus" in loaded_json and isinstance(loaded_json["corpus"], list):
                items = loaded_json["corpus"]
            elif "documents" in loaded_json and isinstance(loaded_json["documents"], list):
                items = loaded_json["documents"]
            else:
                items = [loaded_json]
        else:
            items = [str(loaded_json)]

        for idx, item in enumerate(items):
            if isinstance(item, dict):
                text = str(
                    item.get("text")
                    or item.get("content")
                    or item.get("body")
                    or item.get("passage")
                    or ""
                ).strip()
                if not text:
                    text = json.dumps(item)
                item_title = str(item.get("title") or title or f"{source_name} (Part {idx + 1})")
                item_source = str(item.get("source") or source_name)
                item_id = str(item["id"]) if item.get("id") else f"{source_name}_{idx}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"
                meta = dict(item.get("metadata", {}))
                meta.setdefault("title", item_title)
                meta.setdefault("source", item_source)
                meta.setdefault("chunk_index", idx)
                candidates.append({
                    "id": item_id,
                    "title": item_title,
                    "text": text,
                    "source": item_source,
                    "metadata": meta
                })
            else:
                text = str(item).strip()
                item_title = str(title or f"{source_name} (Part {idx + 1})")
                item_id = f"{source_name}_{idx}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"
                candidates.append({
                    "id": item_id,
                    "title": item_title,
                    "text": text,
                    "source": source_name,
                    "metadata": {"title": item_title, "source": source_name, "chunk_index": idx}
                })

    elif source_name == "user_input":
        # Raw text mode: single document preserved with user title
        text = content.strip()
        doc_title = title or "Custom Knowledge Note"
        cid = f"custom_{hashlib.md5((doc_title + ':' + text).encode('utf-8')).hexdigest()[:10]}"
        candidates.append({
            "id": cid,
            "title": doc_title,
            "text": text,
            "source": "user_input",
            "metadata": {"title": doc_title, "source": "user_input"}
        })

    else:
        # Plain text (.txt, .md, etc.): split by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs and content.strip():
            paragraphs = [content.strip()]

        for idx, p in enumerate(paragraphs):
            cid = f"{source_name}_{idx}_{hashlib.md5(p.encode('utf-8')).hexdigest()[:10]}"
            p_title = f"{source_name} (Part {idx + 1})"
            candidates.append({
                "id": cid,
                "title": p_title,
                "text": p,
                "source": source_name,
                "metadata": {"title": p_title, "source": source_name, "chunk_index": idx}
            })

    # 2. Build fast lookups for existing items
    existing_corpus_by_id: Dict[str, Dict[str, Any]] = {}
    existing_corpus_by_hash: Dict[Tuple[str, str], str] = {}
    for c in corpus:
        if isinstance(c, dict) and "id" in c:
            cid_str = str(c["id"])
            existing_corpus_by_id[cid_str] = c
            c_text = str(c.get("text", "")).strip()
            c_src = str(c.get("source", ""))
            if c_text:
                h = hashlib.md5(c_text.encode("utf-8")).hexdigest()
                existing_corpus_by_hash[(c_src, h)] = cid_str

    vstore_docs: Dict[str, Any] = getattr(vector_store, "documents", {}) if vector_store is not None else {}
    existing_vstore_by_id: Dict[str, Any] = {}
    existing_vstore_by_hash: Dict[Tuple[str, str], str] = {}
    for vid, vdoc in vstore_docs.items():
        if isinstance(vdoc, dict) and "id" in vdoc:
            vid_str = str(vdoc["id"])
            existing_vstore_by_id[vid_str] = vdoc
            v_text = str(vdoc.get("content", "")).strip()
            v_src = str(vdoc.get("source", ""))
            if v_text:
                h = hashlib.md5(v_text.encode("utf-8")).hexdigest()
                existing_vstore_by_hash[(v_src, h)] = vid_str

    added_count = 0
    skipped_count = 0
    chunk_ids: List[str] = []
    new_corpus_items: List[Dict[str, Any]] = []
    new_vstore_chunks: List[DocumentChunk] = []

    for cand in candidates:
        cand_id = cand["id"]
        c_text = cand["text"].strip()
        c_source = cand["source"]
        c_title = cand["title"]
        c_meta = cand["metadata"]
        content_hash = hashlib.md5(c_text.encode("utf-8")).hexdigest()
        source_content_key = (c_source, content_hash)

        # Content-hash deduplication: check if exact (source, content) already exists
        matched_content_id = None
        if source_content_key in existing_corpus_by_hash:
            matched_content_id = existing_corpus_by_hash[source_content_key]
        elif source_content_key in existing_vstore_by_hash:
            matched_content_id = existing_vstore_by_hash[source_content_key]

        if matched_content_id:
            # Document with same source and content exists
            in_corpus = (matched_content_id in existing_corpus_by_id)
            in_vstore = (matched_content_id in existing_vstore_by_id)
            if in_corpus and (in_vstore or vector_store is None):
                # Fully present in both stores: duplicate, safely skip
                skipped_count += 1
                continue
            # Partially present in one store: synchronize missing store using matched_content_id
            target_id = matched_content_id
        else:
            # New content from this source: check for ID collision with different content
            target_id = cand_id
            corpus_collision = False
            if target_id in existing_corpus_by_id:
                if str(existing_corpus_by_id[target_id].get("text", "")).strip() != c_text:
                    corpus_collision = True

            vstore_collision = False
            if target_id in existing_vstore_by_id:
                if str(existing_vstore_by_id[target_id].get("content", "")).strip() != c_text:
                    vstore_collision = True

            if corpus_collision or vstore_collision:
                if on_conflict == "error":
                    raise ValueError(
                        f"Document ID collision: '{target_id}' already exists with different content."
                    )
                # Generate deterministic alternative ID
                alt_base = f"{target_id}_{content_hash[:8]}"
                alt_id = alt_base
                counter = 1
                while (alt_id in existing_corpus_by_id) or (alt_id in existing_vstore_by_id):
                    # If this alt_id has the exact same content, it was already ingested
                    c_same = (alt_id in existing_corpus_by_id and str(existing_corpus_by_id[alt_id].get("text", "")).strip() == c_text)
                    v_same = (alt_id in existing_vstore_by_id and str(existing_vstore_by_id[alt_id].get("content", "")).strip() == c_text)
                    if c_same and (v_same or vector_store is None):
                        target_id = alt_id
                        break
                    alt_id = f"{alt_base}_{counter}"
                    counter += 1
                else:
                    target_id = alt_id

        # Check presence under target_id
        in_corpus = (target_id in existing_corpus_by_id)
        in_vstore = (target_id in existing_vstore_by_id)

        if in_corpus and (in_vstore or vector_store is None):
            c_match = (str(existing_corpus_by_id[target_id].get("text", "")).strip() == c_text)
            v_match = (vector_store is None or str(existing_vstore_by_id[target_id].get("content", "")).strip() == c_text)
            if c_match and v_match:
                skipped_count += 1
                continue

        added_count += 1
        chunk_ids.append(target_id)

        cand_meta = dict(c_meta)
        if target_id != cand_id:
            cand_meta["original_id"] = cand_id

        if not in_corpus:
            corpus_item = {
                "id": target_id,
                "title": c_title,
                "text": c_text,
                "source": c_source,
                "metadata": cand_meta
            }
            corpus.append(corpus_item)
            new_corpus_items.append(corpus_item)
            existing_corpus_by_id[target_id] = corpus_item
            existing_corpus_by_hash[source_content_key] = target_id

        if not in_vstore and vector_store is not None:
            vstore_chunk: DocumentChunk = {
                "id": target_id,
                "content": c_text,
                "source": c_source,
                "score": 0.0,
                "dynamic_weight": 0.0,
                "metadata": dict(cand_meta)
            }
            new_vstore_chunks.append(vstore_chunk)
            existing_vstore_by_id[target_id] = vstore_chunk
            existing_vstore_by_hash[source_content_key] = target_id

    # 3. Synchronize stores
    if new_vstore_chunks and vector_store is not None:
        vector_store.add_documents(new_vstore_chunks)

    if new_corpus_items and retriever is not None:
        retriever.index_corpus(corpus)

    return {
        "added_count": added_count,
        "skipped_count": skipped_count,
        "total_corpus": len(corpus),
        "total_vstore": len(vector_store.documents) if vector_store and hasattr(vector_store, "documents") else 0,
        "chunk_ids": chunk_ids
    }

