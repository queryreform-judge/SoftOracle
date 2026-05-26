from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Optional, Dict, Any

# /mnt/data/son/Thesis/t5/ce_data/umbrella_eval/Qwen2.5-7B-Instruca/trecdl19_methods/0010.trec

# import os
# import random
# from collections import defaultdict

# queries = defaultdict(list)
# original_queries = defaultdict(str)
# datasets = ['2019', '2020', 'dlhard']
# for folder_name in os.listdir('/mnt/data/son/Thesis/t5/llm_conditioned'):
#     if 'Qwen2.5-7B-Instruct' in folder_name:
#         for dataset in datasets:
#             with open(f'/mnt/data/son/Thesis/t5/llm_conditioned/{folder_name}/query2doc_zs/{dataset}/queries/original_queries.tsv', 'r') as f:
#                 for line in f:
#                     qid, text = line.strip().split('\t')
#                     original_queries[qid] = text
#             with open(f'/mnt/data/son/Thesis/t5/llm_conditioned/{folder_name}/query2doc_zs/{dataset}/queries/reformulated_queries.tsv', 'r') as f:
#                 for line in f:
#                     qid, text = line.strip().split('\t')
#                     queries[qid].append(text.replace((original_queries[qid] + ' ') * 5, '').strip())

@register_method("lamer")
class LameR(BaseReformulator):
    VERSION = "0.1"
    REQUIRES_CONTEXT = True
    CONCATENATION_STRATEGY = "interleaved_query_content"  # q + a1 + q + a2 + q + a3 + q + a4 + q + a5

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Use provided contexts (from batch retrieval)
        ctxs = contexts or []

        # Prepare contexts for prompting (use top-10 by default)
        retrieval_k = int(self.cfg.params.get("retrieval_k", 10))
        prompt_ctxs = ctxs[:retrieval_k]
        contexts_blob = "\n".join([f"{i+1}. {psg}" for i, psg in enumerate(prompt_ctxs)])
        # contexts_blob = "\n".join([f"{psg}" for i, psg in enumerate(prompt_ctxs)])

        # Prompt LLM to generate multiple passages given contexts
        gen_passages: List[str] = []
        num_generations = int(self.cfg.params.get("gen_passages", 5))
        # num_generations = 2
        max_tokens = int(self.cfg.llm.get("max_tokens", 128))
        temperature = float(self.cfg.llm.get("temperature", 1.0))
        # llm_generated_passages = random.sample(queries[q.qid], 2)
        # print('llm_generated_passages: ', len(llm_generated_passages))
        # Use prompt template defined in prompt_bank.yaml (lamer.msmarco.v1)
        msgs = self.prompts.render("lamer.msmarco.v1", query=q.text, contexts=contexts_blob)
        # print('msgs: ', msgs)
        for _ in range(num_generations):
            passage = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            # Clean up quotes from the generated passage
            passage = passage.strip().strip('"').strip("'")
            if '**Answering Passage:**' in passage:
                passage = passage.split('**Answering Passage:**')[-1].strip()
            gen_passages.append(passage)
        
        # gen_passages = prompt_ctxs[:num_generations]
        # gen_passages = llm_generated_passages + gen_passages

        # Construct final expanded query using interleaved_query_content
        reformulated = self.concatenate_result(q.text, gen_passages)
        # print('reformulated: ', reformulated)
        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "generated_passages": gen_passages,  # Store the actual passages
                "generated_passages_count": len(gen_passages),
                "used_ctx": len(prompt_ctxs),
            },
        )

    def _get_retrieval_params(self) -> Optional[Dict[str, Any]]:
        """Get LameR-specific retrieval parameters for batch retrieval.
        
        Returns retrieval parameters compatible with BaseSearcher interface.
        Supports both searcher_type/searcher_kwargs format and legacy format.
        """
        # Check if searcher is provided directly (wrappers, adapter instances, etc.)
        if "searcher" in self.cfg.params:
            # Direct searcher instance provided
            return {
                "searcher": self.cfg.params["searcher"],
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }
        
        # Check if using new searcher interface format with searcher_type
        if "searcher_type" in self.cfg.params:
            # New format: use searcher_type and searcher_kwargs
            searcher_type = self.cfg.params.get("searcher_type", "pyserini")
            searcher_kwargs = self.cfg.params.get("searcher_kwargs", {})
            
            # If index is provided in params but not in searcher_kwargs, add it
            if "index" in self.cfg.params and "index" not in searcher_kwargs:
                searcher_kwargs["index"] = self.cfg.params["index"]
            
            return {
                "searcher_type": searcher_type,
                "searcher_kwargs": searcher_kwargs,
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }
        
        # Legacy format: convert old-style params to new format
        index = self.cfg.params.get("index")
        if not index:
            return None
        
        # Build searcher_kwargs from legacy params
        searcher_kwargs = {
            "index": index,
            "searcher_type": "impact" if self.cfg.params.get("impact", False) else "bm25",
            "answer_key": self.cfg.params.get("answer_key", "contents"),
        }
        
        # Add BM25 parameters if specified
        k1 = self.cfg.params.get("k1")
        b = self.cfg.params.get("b")
        if k1 is not None and b is not None:
            searcher_kwargs["k1"] = k1
            searcher_kwargs["b"] = b
        
        # Add RM3 if requested
        if self.cfg.params.get("rm3", False):
            searcher_kwargs["rm3"] = True
        
        # Add Rocchio if requested
        if self.cfg.params.get("rocchio", False):
            searcher_kwargs["rocchio"] = True
            if self.cfg.params.get("rocchio_use_negative", False):
                searcher_kwargs["rocchio_use_negative"] = True
        
        return {
            "searcher_type": "pyserini",  # Default to pyserini for legacy format
            "searcher_kwargs": searcher_kwargs,
            "k": int(self.cfg.params.get("retrieval_k", 10)),
            "threads": int(self.cfg.params.get("threads", 16)),
        }
