import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.llm import LLM
from src.signals import _json, norm

class Generator:
    """
    Synthesizes answers grounded in retrieved passages, extracts citations,
    and conducts multi-draft agentic critique-and-revision with majority voting.
    """
    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Single-pass grounded answer synthesis with evidence citations.
        """
        if not docs:
            prompt = (
                f"Question: {question}\n"
                "Answer the question directly. If you are uncertain, state that you do not know."
            )
            raw_answer = self.llm.call(prompt, temperature=0.0)
            return {"answer": raw_answer.strip(), "citations": []}

        passages_str = "\n".join(
            f"[{i+1}] (ID: {d.get('id', i+1)}, Title: {d.get('title', 'Unknown')}): {d.get('text', '')}"
            for i, d in enumerate(docs)
        )

        prompt = (
            f"Question: {question}\n"
            "Answer using only the passages below. Cite passage numbers like [1], [2] when stating facts.\n"
            "If the passages do not contain the answer, say you cannot find it.\n\n"
            f"Passages:\n{passages_str}\n\n"
            "Answer with citations:"
        )

        ans_text = self.llm.call(prompt, temperature=0.0).strip()

        # Extract citations like [1], [2] and map to document IDs
        citation_nums = [int(m) for m in re.findall(r"\[(\d+)\]", ans_text)]
        cited_ids = []
        for c in citation_nums:
            idx = c - 1
            if 0 <= idx < len(docs):
                cited_ids.append(docs[idx].get("id", str(c)))

        return {
            "answer": ans_text,
            "citations": list(dict.fromkeys(cited_ids))  # unique preserving order
        }

    def critic(self, question: str, draft: str, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Level 3 Agentic Critic: Inspects draft against evidence for factual holes or hallucinations.
        """
        ev = "\n".join(f"[{i+1}] {d.get('text', '')}" for i, d in enumerate(docs))
        prompt = (
            f"Question: {question}\n"
            f"Evidence:\n{ev}\n\n"
            f"Draft Answer: {draft}\n"
            "Check the draft against the passages. List unsupported claims or missing details.\n"
            "Return JSON only: {\"issues\": [\"issue 1\", \"issue 2\"]}"
        )
        raw = self.llm.call(prompt, temperature=0.0)
        parsed = _json(raw, {"issues": []})
        issues = parsed.get("issues", [])
        return issues if isinstance(issues, list) else []

    def revise(self, question: str, draft: str, issues: List[str], docs: List[Dict[str, Any]]) -> str:
        """
        Level 3 Agentic Reviser: Rewrites answer addressing issues raised by the critic.
        """
        if not issues:
            return draft
        issues_str = "\n".join(f"- {issue}" for issue in issues)
        ev = "\n".join(f"[{i+1}] {d.get('text', '')}" for i, d in enumerate(docs))
        prompt = (
            f"Question: {question}\n"
            f"Evidence:\n{ev}\n\n"
            f"Original Draft: {draft}\n"
            f"Identified Issues:\n{issues_str}\n\n"
            "Rewrite the answer to fix these issues using only the passages. Cite passage numbers like [1]."
        )
        revised = self.llm.call(prompt, temperature=0.0).strip()
        return revised

    def answer_level_3(self, question: str, docs: List[Dict[str, Any]], n_drafts: int = 3, max_revisions: int = 2) -> Dict[str, Any]:
        """
        Full agentic Level 3 pipeline:
        N parallel drafts at T=0.7 -> critic -> revise -> majority vote.
        """
        revised_drafts: List[str] = []

        for sample_id in range(n_drafts):
            passages_str = "\n".join(
                f"[{i+1}] {d.get('text', '')}" for i, d in enumerate(docs)
            )
            prompt = (
                f"Question: {question}\n"
                "Answer using only the passages below. Cite passage numbers like [1].\n\n"
                f"Passages:\n{passages_str}"
            )
            draft = self.llm.call(prompt, temperature=0.7, sample_id=sample_id).strip()

            current_draft = draft
            for _ in range(max_revisions):
                issues = self.critic(question, current_draft, docs)
                if not issues:
                    break
                current_draft = self.revise(question, current_draft, issues, docs)

            revised_drafts.append(current_draft)

        # Majority vote across normalized revised drafts
        normed = [norm(d) for d in revised_drafts]
        counts = Counter(normed)
        best_norm = counts.most_common(1)[0][0]

        # Select corresponding original text draft
        final_answer = revised_drafts[0]
        for d in revised_drafts:
            if norm(d) == best_norm:
                final_answer = d
                break

        # Extract citations
        citation_nums = [int(m) for m in re.findall(r"\[(\d+)\]", final_answer)]
        cited_ids = []
        for c in citation_nums:
            idx = c - 1
            if 0 <= idx < len(docs):
                cited_ids.append(docs[idx].get("id", str(c)))

        return {
            "answer": final_answer,
            "citations": list(dict.fromkeys(cited_ids)),
            "all_drafts": revised_drafts
        }

