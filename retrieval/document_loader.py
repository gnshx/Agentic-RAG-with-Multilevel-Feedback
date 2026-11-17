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
        c_source = str(cand["source"]).strip()
        c_title = cand["title"]
        c_meta = cand["metadata"]
        content_hash = hashlib.md5(c_text.encode("utf-8")).hexdigest()
        source_content_key = (c_source, content_hash)

        # 2a. Content + Source Match in both stores
        corpus_matched_id = existing_corpus_by_hash.get(source_content_key)
        vstore_matched_id = existing_vstore_by_hash.get(source_content_key)

        if corpus_matched_id is not None and vstore_matched_id is not None:
            if corpus_matched_id == vstore_matched_id:
                # Same source + same content already in both stores under identical ID -> skip duplicate
                skipped_count += 1
                chunk_ids.append(corpus_matched_id)
                continue
            else:
                # Cross-store ID mismatch for identical source + content!
                # Reconcile safely: choose a canonical ID and align both stores without creating duplicates.
                if cand_id == vstore_matched_id:
                    canonical_id = vstore_matched_id
                    # Collision prevention: check whether canonical_id already belongs to a different corpus document
                    if canonical_id in existing_corpus_by_id and canonical_id != corpus_matched_id:
                        raise ValueError(
                            f"Cannot reconcile document ID: target canonical ID '{canonical_id}' "
                            f"already exists in corpus as a different document."
                        )

                    # Update corpus to match canonical_id with full rollback guard
                    old_corpus_entry = None
                    for c in corpus:
                        if isinstance(c, dict) and c.get("id") == corpus_matched_id:
                            old_corpus_entry = c
                            break
                    if old_corpus_entry is not None:
                        old_corpus_id = old_corpus_entry["id"]

                        # Snapshot retriever internal state before indexing
                        prev_retriever_state = None
                        retriever_shares_corpus = False
                        if retriever is not None:
                            prev_retriever_state = {}
                            retriever_shares_corpus = (
                                hasattr(retriever, "corpus") and retriever.corpus is corpus
                            )
                            if hasattr(retriever, "corpus") and not retriever_shares_corpus:
                                prev_retriever_state["corpus"] = [
                                    dict(doc) if isinstance(doc, dict) else doc
                                    for doc in retriever.corpus
                                ]
                            if hasattr(retriever, "tokenized_corpus") and isinstance(retriever.tokenized_corpus, list):
                                prev_retriever_state["tokenized_corpus"] = [list(toks) for toks in retriever.tokenized_corpus]
                            if hasattr(retriever, "bm25"):
                                prev_retriever_state["bm25"] = getattr(retriever, "bm25", None)
                            if hasattr(retriever, "dense_vectors") and isinstance(retriever.dense_vectors, list):
                                prev_retriever_state["dense_vectors"] = [
                                    vec.copy() if hasattr(vec, "copy") else vec
                                    for vec in retriever.dense_vectors
                                ]

                        old_corpus_entry["id"] = canonical_id
                        try:
                            if retriever is not None:
                                retriever.index_corpus(corpus)
                        except Exception as e:
                            # 1. Rollback corpus entry ID
                            old_corpus_entry["id"] = old_corpus_id
                            # 2. Rollback retriever internal state to pre-indexing snapshot while preserving shared list identity
                            if retriever is not None:
                                if retriever_shares_corpus:
                                    retriever.corpus = corpus
                                elif prev_retriever_state and "corpus" in prev_retriever_state:
                                    retriever.corpus = prev_retriever_state["corpus"]
                                if prev_retriever_state:
                                    for attr in ("tokenized_corpus", "bm25", "dense_vectors"):
                                        if attr in prev_retriever_state:
                                            setattr(retriever, attr, prev_retriever_state[attr])
                            raise RuntimeError(
                                f"Failed to reconcile corpus ID to '{canonical_id}': {e}"
                            ) from e
                        existing_corpus_by_id.pop(corpus_matched_id, None)
                        existing_corpus_by_id[canonical_id] = old_corpus_entry
                        existing_corpus_by_hash[source_content_key] = canonical_id
                else:
                    canonical_id = corpus_matched_id
                    # Update vector_store to match canonical_id
                    if vector_store is not None:
                        if hasattr(vector_store, "reconcile_id"):
                            vector_store.reconcile_id(vstore_matched_id, canonical_id)
                        else:
                            # In-memory fallback if mock vector store without reconcile_id
                            if hasattr(vector_store, "documents") and vstore_matched_id in vector_store.documents:
                                old_chunk = vector_store.documents[vstore_matched_id]
                                try:
                                    vector_store.documents.pop(vstore_matched_id)
                                    old_chunk["id"] = canonical_id
                                    vector_store.documents[canonical_id] = old_chunk
                                    if hasattr(vector_store, "vectors") and vstore_matched_id in vector_store.vectors:
                                        vector_store.vectors[canonical_id] = vector_store.vectors.pop(vstore_matched_id)
                                except Exception as err:
                                    old_chunk["id"] = vstore_matched_id
                                    vector_store.documents[vstore_matched_id] = old_chunk
                                    vector_store.documents.pop(canonical_id, None)
                                    raise RuntimeError(f"Fallback reconciliation failed: {err}") from err

                    existing_vstore_by_id.pop(vstore_matched_id, None)
                    if vector_store is not None and hasattr(vector_store, "documents") and canonical_id in vector_store.documents:
                        existing_vstore_by_id[canonical_id] = vector_store.documents[canonical_id]
                    existing_vstore_by_hash[source_content_key] = canonical_id

                skipped_count += 1
                chunk_ids.append(canonical_id)
                continue

        elif corpus_matched_id is not None and vstore_matched_id is None:
            # Document present in corpus, missing from vector store -> synchronize into vector_store
            target_id = corpus_matched_id
            if vector_store is not None:
                vstore_chunk: DocumentChunk = {
                    "id": target_id,
                    "content": c_text,
                    "source": c_source,
                    "score": 0.0,
                    "dynamic_weight": 0.0,
                    "metadata": dict(c_meta)
                }
                new_vstore_chunks.append(vstore_chunk)
                existing_vstore_by_id[target_id] = vstore_chunk
                existing_vstore_by_hash[source_content_key] = target_id
            added_count += 1
            chunk_ids.append(target_id)
            continue

        elif corpus_matched_id is None and vstore_matched_id is not None:
            # Document present in vector store, missing from corpus -> synchronize into corpus
            target_id = vstore_matched_id
            corpus_item = {
                "id": target_id,
                "title": c_title,
                "text": c_text,
                "source": c_source,
                "metadata": c_meta
            }
            corpus.append(corpus_item)
            new_corpus_items.append(corpus_item)
            existing_corpus_by_id[target_id] = corpus_item
            existing_corpus_by_hash[source_content_key] = target_id
            added_count += 1
            chunk_ids.append(target_id)
            continue

        # 2b. Content + Source NOT found in either store under any ID:
        # Check whether cand_id collides with an existing document with different content OR different source
        target_id = cand_id
        corpus_collision = False
        if target_id in existing_corpus_by_id:
            existing_c = existing_corpus_by_id[target_id]
            if str(existing_c.get("text", "")).strip() != c_text or str(existing_c.get("source", "")).strip() != c_source:
                corpus_collision = True

        vstore_collision = False
        if target_id in existing_vstore_by_id:
            existing_v = existing_vstore_by_id[target_id]
            if str(existing_v.get("content", "")).strip() != c_text or str(existing_v.get("source", "")).strip() != c_source:
                vstore_collision = True

        if corpus_collision or vstore_collision:
            if on_conflict == "error":
                raise ValueError(
                    f"Document ID collision: '{target_id}' already exists with different content or source."
                )
            # Generate deterministic alternative ID based on source and content hash
            src_cnt_hash = hashlib.md5(f"{c_source}:{c_text}".encode("utf-8")).hexdigest()
            alt_base = f"{target_id}_{src_cnt_hash[:8]}"
            alt_id = alt_base
            counter = 1
            while (alt_id in existing_corpus_by_id) or (alt_id in existing_vstore_by_id):
                # Check if this alt_id matches BOTH content AND source
                c_same = (
                    alt_id in existing_corpus_by_id
                    and str(existing_corpus_by_id[alt_id].get("text", "")).strip() == c_text
                    and str(existing_corpus_by_id[alt_id].get("source", "")).strip() == c_source
                )
                v_same = (
                    alt_id in existing_vstore_by_id
                    and str(existing_vstore_by_id[alt_id].get("content", "")).strip() == c_text
                    and str(existing_vstore_by_id[alt_id].get("source", "")).strip() == c_source
                )
                if c_same and (v_same or vector_store is None):
                    target_id = alt_id
                    break
                alt_id = f"{alt_base}_{counter}"
                counter += 1
            else:
                target_id = alt_id

        # 2c. Check presence under target_id (must match BOTH content AND source to be skipped)
        in_corpus = (target_id in existing_corpus_by_id)
        in_vstore = (target_id in existing_vstore_by_id)

        if in_corpus and (in_vstore or vector_store is None):
            c_match = (
                str(existing_corpus_by_id[target_id].get("text", "")).strip() == c_text
                and str(existing_corpus_by_id[target_id].get("source", "")).strip() == c_source
            )
            v_match = (
                vector_store is None
                or (
                    str(existing_vstore_by_id[target_id].get("content", "")).strip() == c_text
                    and str(existing_vstore_by_id[target_id].get("source", "")).strip() == c_source
                )
            )
            if c_match and v_match:
                skipped_count += 1
                chunk_ids.append(target_id)
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

