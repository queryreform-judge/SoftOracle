"""
CSQE Modified: same pipeline and prompts as CSQE, but uses only judge-selected
documents as the top-k context (instead of raw retrieval top-k).

Configure via params:
  - judge_docids_map: Dict[qid, List[docid]] (precomputed), or
  - qrel_paths: List[str] of .qrel paths to load judge mapping from.
  - judge_rel_mode: "positive" | "negative" | "both" — which qrel rows to include:
      - "positive": only rel > 0 (relevant)
      - "negative": only rel == 0 (non-relevant)
      - "both": rel > 0 and rel == 0; {contexts} is formatted as
          "Relevant documents:\n1. ...\n\nNon relevant documents:\n1. ..." so the prompt can distinguish.

When ctx_map provides (docid, content) pairs per query, only docs in the judge
list for that query are used; then the first retrieval_k of those are passed
to the same CSQE pipeline. Plain list-of-strings contexts work as in CSQE (no filtering).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.base import QueryItem, ReformulationResult
from ..core.registry import register_method
from .csqe import CSQE
import random

def load_judge_mapping(
    qrel_paths: Union[str, Path, List[Union[str, Path]]],
    *,
    four_col_rel_idx: int = 3,
    six_col_rel_idx: int = 3,
    rel_mode: str = "positive",
) -> Dict[str, List[str]]:
    """Load qid -> list of doc IDs from one or more qrel files (4-column) or trec files (6-column).

    For .qrel files (4 columns: qid Q0 docid rel):
        rel_mode: which rows to include in the judge list:
            - "positive": only rel > 0 (relevant docs)
            - "negative": only rel == 0 (non-relevant docs)
            - "both": rel > 0 or rel == 0 (all judged docs)
    
    For .trec files (6 columns: qid Q0 docid rank score run_name):
        All docids are included in order (rel_mode is ignored for .trec files).
    """
    if isinstance(qrel_paths, (str, Path)):
        qrel_paths = [qrel_paths]
    mode = (rel_mode or "positive").strip().lower()
    if mode not in ("positive", "negative", "both"):
        mode = "positive"
    mapping: Dict[str, List[str]] = defaultdict(list)
    
    for p in qrel_paths:
        p = Path(p)
        if not p.exists():
            continue
        
        # Detect file format by checking first line
        is_trec_format = False
        with open(p, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                # .trec files have 6 columns, .qrel files have 4 columns
                if len(parts) == 6:
                    is_trec_format = True
        
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                qid, _, docid = parts[0], parts[1], parts[2]
                
                if is_trec_format:
                    # .trec format: qid Q0 docid rank score run_name
                    # Include all docids in order (rel_mode is ignored)
                    mapping[qid].append(docid)
                    random.shuffle(mapping[qid]) # shuffle

                else:
                    # .qrel format: qid Q0 docid rel
                    try:
                        rel = int(parts[four_col_rel_idx])
                    except ValueError:
                        rel = 0
                    
                    if mode == "positive" and rel > 1:
                        print(f"Adding {docid} to {qid} for positive mode")
                        mapping[qid].append(docid)
                    elif mode == "negative" and rel <= 1:
                        mapping[qid].append(docid)
                    elif mode == "both" and (rel > 0 or rel == 0):
                        mapping[qid].append(docid)
    
    return dict(mapping)


def build_ctx_map_with_judge(
    query_ids: List[str],
    batch_results: List[List[Any]],
    judge_docids_map: Dict[str, List[str]],
    retrieval_k: int,
    *,
    docid_attr: str = "docid",
    content_attr: str = "content",
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build ctx_map (qid -> [(docid, content), ...]) from batch search results,
    keeping only judge-relevant docs per query, ordered by judge list.

    Use for csqe_modified when passing pre-retrieved hits so only judge-selected
    docs are used as context.

    Args:
        query_ids: List of query IDs in same order as batch_results.
        batch_results: List of lists of search hits (e.g. SearchHit with .docid, .content).
        judge_docids_map: qid -> list of relevant doc IDs (from load_judge_mapping).
        retrieval_k: Max number of contexts per query.
        docid_attr: Attribute name for doc id on each hit.
        content_attr: Attribute name for content on each hit.

    Returns:
        ctx_map: qid -> [(docid, content), ...] for reformulate_batch(queries, ctx_map).
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for qid, hits in zip(query_ids, batch_results):
        judge_list = judge_docids_map.get(qid)
        if not judge_list:
            out[qid] = []
            continue
        judge_set = set(judge_list)
        order = {d: i for i, d in enumerate(judge_list)}
        pairs: List[Tuple[str, str]] = []
        for h in hits:
            docid = getattr(h, docid_attr, None) if hasattr(h, docid_attr) else (h.get(docid_attr) if isinstance(h, dict) else None)
            content = getattr(h, content_attr, "") if hasattr(h, content_attr) else (h.get(content_attr, "") if isinstance(h, dict) else "")
            if docid is not None and docid in judge_set:
                pairs.append((str(docid), str(content)))
        pairs.sort(key=lambda x: order.get(x[0], len(order)))
        out[qid] = pairs[:retrieval_k]
    return out


def _contexts_to_strings(
    contexts: Union[List[str], List[Tuple[str, str]]],
    judge_docids: Optional[List[str]],
    retrieval_k: int,
) -> List[str]:
    """Return top-k context strings; if (docid, content) and judge_docids set, filter by judge first."""
    if not contexts:
        return []
    judge_set = set(judge_docids) if judge_docids else None
    order_map = {d: i for i, d in enumerate(judge_docids)} if judge_docids else None

    if contexts and isinstance(contexts[0], (list, tuple)) and len(contexts[0]) >= 2:
        pairs = [(c[0], c[1]) for c in contexts if len(c) >= 2]
        if judge_set is not None:
            filtered = [(docid, content) for docid, content in pairs if docid in judge_set]
            if order_map is not None:
                filtered.sort(key=lambda x: order_map.get(x[0], len(order_map)))
            pairs = filtered
        return [content for _, content in pairs[:retrieval_k]]
    return list(contexts)[:retrieval_k]


def _contexts_to_strings_split(
    contexts: Union[List[str], List[Tuple[str, str]]],
    positive_docids: Optional[List[str]],
    negative_docids: Optional[List[str]],
    retrieval_k: int,
) -> str:
    """Format contexts as 'Relevant documents:\\n1. ...\\n\\nNon relevant documents:\\n1. ...' for judge_rel_mode='both'.
    Requires (docid, content) pairs; positive/negative_docids identify which are relevant vs non-relevant.
    """
    relevant: List[str] = []
    non_relevant: List[str] = []
    pos_set = set(positive_docids) if positive_docids else set()
    neg_set = set(negative_docids) if negative_docids else set()
    pos_order = {d: i for i, d in enumerate(positive_docids)} if positive_docids else {}
    neg_order = {d: i for i, d in enumerate(negative_docids)} if negative_docids else {}

    if not contexts:
        pass
    elif contexts and isinstance(contexts[0], (list, tuple)) and len(contexts[0]) >= 2:
        pairs = [(c[0], c[1]) for c in contexts if len(c) >= 2]
        for docid, content in pairs:
            if docid in pos_set:
                relevant.append((docid, content))
            elif docid in neg_set:
                non_relevant.append((docid, content))
        relevant.sort(key=lambda x: pos_order.get(x[0], len(pos_order)))
        non_relevant.sort(key=lambda x: neg_order.get(x[0], len(neg_order)))
        relevant = [c for _, c in relevant[:retrieval_k]]
        non_relevant = [c for _, c in non_relevant[:retrieval_k]]
    else:
        # Plain strings: treat all as relevant, no non-relevant
        relevant = list(contexts)[:retrieval_k]
        non_relevant = []

    lines = []
    lines.append("Relevant documents:")
    for i, c in enumerate(relevant, 1):
        lines.append(f"{i}. {c}")
    lines.append("")
    
    if non_relevant:
        lines.append("Non relevant documents:")
        for i, c in enumerate(non_relevant, 1): 
            lines.append(f"{i}. {c}")

    return "\n".join(lines)


@register_method("csqe_modified")
class CSQEModified(CSQE):
    """
    Same pipeline and prompts as CSQE; only the selected top-k documents differ.

    Uses judge-relevant doc IDs (from qrel_paths or judge_docids_map in params) to
    filter contexts: when contexts are (docid, content) pairs, only docs in the
    judge list for that query are used, then the first retrieval_k are passed to
    KEQE+CSQE. Otherwise behaves like CSQE (plain list of strings -> first retrieval_k).
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = True

    def __init__(self, cfg: Any, llm_client: Any, prompt_resolver: Any):
        super().__init__(cfg, llm_client, prompt_resolver)
        params = getattr(cfg, "params", None) or {}
        self._judge_docids_map = params.get("judge_docids_map")
        self._judge_docids_map_positive: Optional[Dict[str, List[str]]] = None
        self._judge_docids_map_negative: Optional[Dict[str, List[str]]] = None
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
