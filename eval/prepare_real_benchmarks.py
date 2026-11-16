import json
import os
import hashlib
from typing import Dict, Any, List, Tuple

def generate_doc_id(title: str, text: str) -> str:
    """Generate a deterministic, collision-resistant document ID based on title and content."""
    content_hash = hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"doc_{content_hash}"

def clean_sentence_join(sentences: Any) -> str:
    """Clean and join sentence tokens into a well-formed paragraph string."""
    if isinstance(sentences, list):
        return " ".join(s.strip() for s in sentences if s and s.strip())
    return str(sentences).strip()

def build_real_benchmarks(
    n_hotpot_dev: int = 100,
    n_hotpot_test: int = 100,
    n_2wiki_test: int = 100,
    output_dir: str = "data"
) -> Dict[str, Any]:
    """
    Extracts real HotpotQA and 2WikiMultiHopQA validation benchmarks
    and constructs a unified, verified real Wikipedia corpus.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Hugging Face 'datasets' package is required to download real benchmark splits. "
            "Please install it with: pip install datasets"
        )

    print("=== Extracting Genuine HotpotQA & 2WikiMultiHopQA Benchmarks ===")
    os.makedirs(output_dir, exist_ok=True)

    corpus_map: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------
    # 1. HotpotQA Official Validation Split
    # ---------------------------------------------------------
    print("Loading official hotpotqa/hotpot_qa validation split from local datasets cache...")
    hotpot_ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    total_hp = len(hotpot_ds)
    print(f"Loaded HotpotQA validation split: {total_hp} items available.")

    def extract_hotpot_records(slice_obj, split_name: str) -> List[Dict[str, Any]]:
        records = []
        for row in slice_obj:
            q_id = row["id"]
            question = row["question"]
            answer = row["answer"]
            q_type = row.get("type", "bridge")
            level = row.get("level", "medium")

            supporting_titles = set(row["supporting_facts"]["title"])
            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]

            gold_paragraphs = []
            distractor_paragraphs = []
            context_paragraphs = []
            doc_ids = []

            for t, s_list in zip(titles, sentences):
                text = clean_sentence_join(s_list)
                doc_id = generate_doc_id(t, text)
                doc_obj = {
                    "id": doc_id,
                    "title": t,
                    "text": text,
                    "dataset": "HotpotQA"
                }

                if doc_id not in corpus_map:
                    corpus_map[doc_id] = doc_obj

                doc_ids.append(doc_id)
                context_paragraphs.append(doc_obj)
                if t in supporting_titles:
                    gold_paragraphs.append(doc_obj)
                else:
                    distractor_paragraphs.append(doc_obj)

            records.append({
                "id": q_id,
                "dataset": "HotpotQA",
                "type": q_type,
                "difficulty": level,
                "question": question,
                "answer": answer,
                "gold_paragraphs": gold_paragraphs,
                "distractor_paragraphs": distractor_paragraphs,
                "context": context_paragraphs,
                "doc_ids": doc_ids,
                "supporting_titles": list(supporting_titles),
                "split": split_name
            })
        return records

    print(f"Extracting {n_hotpot_dev} HotpotQA Dev questions (0:{n_hotpot_dev})...")
    dev_records = extract_hotpot_records(hotpot_ds.select(range(0, n_hotpot_dev)), "dev")

    print(f"Extracting {n_hotpot_test} HotpotQA Test questions ({n_hotpot_dev}:{n_hotpot_dev + n_hotpot_test})...")
    test_records = extract_hotpot_records(hotpot_ds.select(range(n_hotpot_dev, n_hotpot_dev + n_hotpot_test)), "test")

    # ---------------------------------------------------------
    # 2. 2WikiMultiHopQA Official Validation Split
    # ---------------------------------------------------------
    print("Loading official voidful/2wikimultihopqa validation split from local datasets cache...")
    wiki2_ds = load_dataset("voidful/2wikimultihopqa", split="validation")
    total_wiki = len(wiki2_ds)
    print(f"Loaded 2WikiMultiHopQA validation split: {total_wiki} items available.")

    wiki2_records = []
    for row in wiki2_ds.select(range(0, n_2wiki_test)):
        q_id = row["_id"]
        question = row["question"]
        answer = row["answer"]
        q_type = row.get("type", "compositional")

        supporting_titles = set(sf[0] for sf in row.get("supporting_facts", []))

        gold_paragraphs = []
        distractor_paragraphs = []
        context_paragraphs = []
        doc_ids = []

        for ctx_item in row["context"]:
            t = ctx_item[0]
            s_list = ctx_item[1]
            text = clean_sentence_join(s_list)
            doc_id = generate_doc_id(t, text)
            doc_obj = {
                "id": doc_id,
                "title": t,
                "text": text,
                "dataset": "2WikiMultiHopQA"
            }

            if doc_id not in corpus_map:
                corpus_map[doc_id] = doc_obj

            doc_ids.append(doc_id)
            context_paragraphs.append(doc_obj)
            if t in supporting_titles:
                gold_paragraphs.append(doc_obj)
            else:
                distractor_paragraphs.append(doc_obj)

        wiki2_records.append({
            "id": q_id,
            "dataset": "2WikiMultiHopQA",
            "type": q_type,
            "difficulty": "multi-hop",
            "question": question,
            "answer": answer,
            "gold_paragraphs": gold_paragraphs,
            "distractor_paragraphs": distractor_paragraphs,
            "context": context_paragraphs,
            "doc_ids": doc_ids,
            "supporting_titles": list(supporting_titles),
            "split": "transfer_test"
        })

    # ---------------------------------------------------------
    # 3. Build & Verify Master Real Corpus
    # ---------------------------------------------------------
    corpus_list = list(corpus_map.values())
    unique_doc_count = len(corpus_list)

    # Verification checks
    assert len(dev_records) == n_hotpot_dev, f"Expected {n_hotpot_dev} dev records, got {len(dev_records)}"
    assert len(test_records) == n_hotpot_test, f"Expected {n_hotpot_test} test records, got {len(test_records)}"
    assert len(wiki2_records) == n_2wiki_test, f"Expected {n_2wiki_test} 2Wiki records, got {len(wiki2_records)}"
    assert unique_doc_count > 1000, f"Corpus size must be >1000 paragraphs, found {unique_doc_count}"

    # Save datasets to disk
    dev_path = os.path.join(output_dir, "real_dev.json")
    test_path = os.path.join(output_dir, "real_test.json")
    wiki_path = os.path.join(output_dir, "real_2wiki_test.json")
    corpus_path = os.path.join(output_dir, "real_corpus.json")

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_records, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)
    with open(wiki_path, "w", encoding="utf-8") as f:
        json.dump(wiki2_records, f, indent=2)
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_list, f, indent=2)

    total_words = sum(len(d["text"].split()) for d in corpus_list)
    avg_words = total_words / max(1, unique_doc_count)

    print("\n[SUCCESS] Real Benchmarks & Corpus Created & Verified:")
    print(f"  • {dev_path}: {len(dev_records)} questions (HotpotQA Dev)")
    print(f"  • {test_path}: {len(test_records)} questions (HotpotQA In-Domain Test)")
    print(f"  • {wiki_path}: {len(wiki2_records)} questions (2Wiki Cross-Dataset Transfer)")
    print(f"  • {corpus_path}: {unique_doc_count} unique Wikipedia paragraphs (>1,000 verified!)")
    print(f"    - Total words: {total_words:,}")
    print(f"    - Average passage length: {avg_words:.1f} words/passage")

    return {
        "dev_questions": len(dev_records),
        "test_questions": len(test_records),
        "wiki_questions": len(wiki2_records),
        "total_questions": len(dev_records) + len(test_records) + len(wiki2_records),
        "corpus_paragraphs": unique_doc_count,
        "total_words": total_words,
        "avg_words_per_passage": round(avg_words, 1),
        "dev_path": dev_path,
        "test_path": test_path,
        "wiki_path": wiki_path,
        "corpus_path": corpus_path
    }

if __name__ == "__main__":
    build_real_benchmarks()
