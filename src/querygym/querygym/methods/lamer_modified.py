"""
LameR Modified: same pipeline and prompts as LameR, but uses only judge-selected
documents as the top-k context (instead of raw retrieval top-k).

Configure via params:
  - judge_docids_map: Dict[qid, List[docid]] (precomputed), or
  - qrel_paths: List[str] of .qrel paths to load judge mapping from.
  - judge_rel_mode: "positive" | "negative" | "both" — which qrel rows to include
      (rel > 0, rel == 0, or both). Default "positive".

When ctx_map provides (docid, content) pairs per query, only docs in the judge
list for that query are used; then the first retrieval_k of those are passed
to LameR. Plain list-of-strings contexts work as in LameR (no filtering).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from ..core.base import QueryItem, ReformulationResult
from ..core.registry import register_method
from .lamer import LameR
from .csqe_modified import (
    load_judge_mapping,
    _contexts_to_strings,
    _contexts_to_strings_split,
)


@register_method("lamer_modified")
class LameRModified(LameR):
    """
    Same pipeline and prompts as LameR; only the selected top-k documents differ.

    Uses judge-relevant doc IDs (from qrel_paths or judge_docids_map in params) to
    filter contexts. When judge_rel_mode="both", {contexts} is formatted as
    "Relevant documents:\\n...\\n\\nNon relevant documents:\\n..." so the prompt can distinguish.
    """

    VERSION = "0.1"
    REQUIRES_CONTEXT = True
    CONCATENATION_STRATEGY = "interleaved_query_content"

    def __init__(self, cfg: Any, llm_client: Any, prompt_resolver: Any):
        super().__init__(cfg, llm_client, prompt_resolver)
        params = getattr(cfg, "params", None) or {}
        self._judge_docids_map = params.get("judge_docids_map")
        self._judge_docids_map_positive: Optional[Any] = None
        self._judge_docids_map_negative: Optional[Any] = None
        if self._judge_docids_map is None and params.get("qrel_paths"):
            paths = params["qrel_paths"]
            if isinstance(paths, (str, Path)):
                paths = [paths]
            rel_mode = (params.get("judge_rel_mode") or "positive").strip().lower()
            if rel_mode == "both":
                self._judge_docids_map_positive = load_judge_mapping(paths, rel_mode="positive")
                self._judge_docids_map_negative = load_judge_mapping(paths, rel_mode="negative")
                self._judge_docids_map = None
            else:
                self._judge_docids_map = load_judge_mapping(paths, rel_mode=rel_mode)

    def reformulate(
        self,
        q: QueryItem,
        contexts: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
    ) -> ReformulationResult:
        ctxs = contexts or []
        retrieval_k = int(self.cfg.params.get("retrieval_k", 10))
        if self._judge_docids_map_positive is not None and self._judge_docids_map_negative is not None:
            positive_docids = self._judge_docids_map_positive.get(q.qid)
            negative_docids = self._judge_docids_map_negative.get(q.qid)
            formatted_blob = _contexts_to_strings_split(
                ctxs, positive_docids, negative_docids, retrieval_k
            )
            prompt_ctxs = [formatted_blob]
        else:
            judge_docids = self._judge_docids_map.get(q.qid) if self._judge_docids_map else None

            prompt_ctxs = _contexts_to_strings(ctxs, judge_docids, retrieval_k)
        return super().reformulate(q, prompt_ctxs)
