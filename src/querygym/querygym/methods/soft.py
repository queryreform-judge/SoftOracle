"""
SOFT method: same pipeline as CSQE but uses ksoft.v1 and csoft.v1 prompts
instead of keqe.v1 and csqe.v1.
"""

from __future__ import annotations

from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Optional, Dict, Any

from .csqe import CSQE


@register_method("soft")
class Soft(CSQE):
    """
    SOFT: same flow as CSQE (knowledge expansion + context-based expansion)
    but uses prompts ksoft.v1 (knowledge) and csoft.v1 (context) instead of
    keqe.v1 and csqe.v1.
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = True

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Reformulate using SOFT prompts (ksoft.v1, csoft.v1)."""
        ctxs = contexts or []
        retrieval_k = int(self.cfg.params.get("retrieval_k", 10))
        prompt_ctxs = ctxs[:retrieval_k]
        contexts_blob = "\n".join([f"{i+1}. {psg}" for i, psg in enumerate(prompt_ctxs)])

        n_expansions = int(self.cfg.params.get("gen_num", 2))
        max_tokens = int(self.cfg.llm.get("max_tokens", 1024))
        temperature = float(self.cfg.llm.get("temperature", 1.0))

        # Step 1: Knowledge expansion with ksoft.v1 (instead of keqe.v1)
        msgs_ksoft = self.prompts.render("ksoft.v1", query=q.text)
        resp_ksoft = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=msgs_ksoft,
            n=n_expansions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ksoft_passages = [
            choice.message.content.strip().strip('"').strip("'") or ""
            for choice in resp_ksoft.choices
        ]

        # Step 2: Context expansion with csoft.v1 (instead of csqe.v1)
        msgs_csoft = self.prompts.render("csoft.v1", query=q.text, contexts=contexts_blob)
        resp_csoft = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=msgs_csoft,
            n=n_expansions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        csoft_responses = [choice.message.content or "" for choice in resp_csoft.choices]

        # Step 3: Extract key sentences (same as CSQE)
        print(csoft_responses)
        csoft_sentences: List[str] = []
        for response in csoft_responses:
            extracted = self._extract_key_sentences(response)
            csoft_sentences.append(extracted)
        print('csoft_sentences: ', csoft_sentences)
        # Step 4: Concatenate (query × N + ksoft_passages + csoft_sentences)
        parts = [q.text] * n_expansions + csoft_sentences
        reformulated = " ".join(parts).lower().strip().strip('"').strip("'")

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "ksoft_passages": ksoft_passages,
                "csoft_responses": csoft_responses,
                "csoft_sentences": csoft_sentences,
                "gen_num": n_expansions,
                "total_generations": n_expansions * 2,
                "used_ctx": len(prompt_ctxs),
            },
        )
