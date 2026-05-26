"""
Pairwise approach for selecting the best query variant based on retrieved documents.

Given a query and multiple variants (each with top-k retrieved documents),
compare all pairs of variants using an LLM judge to determine which result set
better satisfies the user's information need.

The variant with the most wins is selected for each query.
"""

import argparse
import os
import re
import json
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from umbrela.gpt_judge import GPTJudge
from umbrela.gemini_judge import GeminiJudge
from umbrela.os_llm_judge import OSLLMJudge
from umbrela.utils import qrel_utils


@dataclass
class VariantResult:
    """Represents a variant's retrieval results for a query."""
    variant_id: str
    doc_ids: List[str]
    doc_texts: List[str]
    scores: List[float]


PAIRWISE_PROMPT_TEMPLATE = """You are an expert search quality evaluator. Your task is to compare two sets of search results and determine which one better satisfies the user's information need.

Query: {query}

=== Result Set A ===
{result_set_a}

=== Result Set B ===
{result_set_b}

Instructions:
1. Consider how well each result set addresses the user's query
2. Evaluate relevance, coverage, and quality of the documents
3. Think step-by-step about which set is more helpful

After your analysis, provide your final answer on a new line in EXACTLY this format:
WINNER: A
or
WINNER: B

Your response:"""


def load_trec_run(run_file_path: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """
    Reads a TREC run file and returns a dict: {qid: [(docid, score), ...]}
    Limit to top_k docs per query.
    """
    run_data = defaultdict(list)
    with open(run_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            # Standard TREC format: qid Q0 docid rank score runtag
            qid = parts[0]
            docid = parts[2]
            score = float(parts[4])
            
            run_data[qid].append((docid, score))
    
    # Sort by score and limit to top_k
    for qid in run_data:
        run_data[qid] = sorted(run_data[qid], key=lambda x: x[1], reverse=True)[:top_k]
    
    return dict(run_data)


def format_result_set(docs: List[Tuple[str, str]], max_chars_per_doc: int = 500) -> str:
    """
    Format a list of (doc_id, doc_text) tuples into a readable string.
    Truncates long documents to save tokens.
    """
    formatted = []
    for i, (doc_id, doc_text) in enumerate(docs, 1):
        truncated_text = doc_text[:max_chars_per_doc]
        if len(doc_text) > max_chars_per_doc:
            truncated_text += "..."
        formatted.append(f"[Doc {i}] {truncated_text}")
    return "\n\n".join(formatted)


def parse_pairwise_response(response: str) -> Optional[str]:
    """
    Parse the LLM response to extract the winner (A or B).
    Returns None if parsing fails.
    """
    response = response.strip().upper()
    
    # Try to find "WINNER: A" or "WINNER: B" pattern
    patterns = [
        r'WINNER:\s*([AB])',
        r'WINNER\s*=\s*([AB])',
        r'WINNER\s+IS\s+([AB])',
        r'THE\s+WINNER\s+IS\s+([AB])',
        r'RESULT\s+SET\s+([AB])\s+(?:IS\s+)?(?:THE\s+)?(?:BETTER|WINNER)',
        r'([AB])\s+IS\s+(?:THE\s+)?(?:BETTER|WINNER)',
        r'\b([AB])\s*$',  # Just A or B at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback: count mentions of A vs B in positive context
    a_count = len(re.findall(r'\b(?:SET\s+)?A\s+(?:IS\s+)?(?:BETTER|MORE|SUPERIOR)', response))
    b_count = len(re.findall(r'\b(?:SET\s+)?B\s+(?:IS\s+)?(?:BETTER|MORE|SUPERIOR)', response))
    
    if a_count > b_count:
        return "A"
    elif b_count > a_count:
        return "B"
    
    return None


class PairwiseJudge:
    """
    Pairwise judge for comparing query variant results using an LLM.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        vllm_port: int = 1234,
        vllm_host: str = "localhost",
        use_os_llm: bool = False,
        max_new_tokens: int = 500,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_new_tokens = max_new_tokens
        
        # Initialize the appropriate judge backend
        if use_os_llm or any(keyword in model_name.lower() for keyword in ['qwen', 'llama', 'mistral', 'phi', 'gemma']):
            self.judge = OSLLMJudge(
                qrel=dataset_name,
                model_name=model_name,
                prompt_type="bing",
                few_shot_count=0,
                vllm_port=vllm_port,
                vllm_host=vllm_host,
            )
            self.judge_type = "os_llm"
        elif 'gpt' in model_name.lower():
            self.judge = GPTJudge(
                qrel=dataset_name,
                model_name=model_name,
                prompt_type="bing",
                few_shot_count=0,
            )
            self.judge_type = "gpt"
        else:
            self.judge = GeminiJudge(
                qrel=dataset_name,
                model_name=model_name,
                prompt_type="bing",
                few_shot_count=0,
            )
            self.judge_type = "gemini"
    
    def _run_llm(self, prompt: str) -> str:
        """Run the appropriate LLM based on judge type."""
        if self.judge_type == "os_llm":
            return self.judge.run_os_llm(prompt, max_new_tokens=self.max_new_tokens)
        elif self.judge_type == "gpt":
            return self.judge.run_gpt(prompt, max_new_tokens=self.max_new_tokens)
        else:
            return self.judge.run_gemini(prompt, max_new_tokens=self.max_new_tokens)
    
    def compare_pair(
        self,
        query: str,
        docs_a: List[Tuple[str, str]],
        docs_b: List[Tuple[str, str]],
    ) -> Tuple[str, str]:
        """
        Compare two result sets and return the winner.
        
        Args:
            query: The search query
            docs_a: List of (doc_id, doc_text) for result set A
            docs_b: List of (doc_id, doc_text) for result set B
            
        Returns:
            Tuple of (winner: "A" or "B", raw_response)
        """
        result_set_a = format_result_set(docs_a)
        result_set_b = format_result_set(docs_b)
        
        prompt = PAIRWISE_PROMPT_TEMPLATE.format(
            query=query,
            result_set_a=result_set_a,
            result_set_b=result_set_b,
        )
        
        response = self._run_llm(prompt)
        winner = parse_pairwise_response(response)
        
        if winner is None:
            # Default to A if parsing fails
            print(f"Warning: Could not parse winner from response. Defaulting to A.")
            winner = "A"
        
        return winner, response


def pairwise_selection(
    query: str,
    variants: List[VariantResult],
    judge: PairwiseJudge,
    verbose: bool = False,
) -> Tuple[int, Dict[int, int]]:
    """
    Perform pairwise tournament selection to find the best variant.
    
    Args:
        query: The search query
        variants: List of VariantResult objects
        judge: PairwiseJudge instance
        verbose: Whether to print detailed comparison info
        
    Returns:
        Tuple of (best_variant_index, wins_dict)
    """
    n = len(variants)
    wins = {i: 0 for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            # Prepare document lists
            docs_a = list(zip(variants[i].doc_ids, variants[i].doc_texts))
            docs_b = list(zip(variants[j].doc_ids, variants[j].doc_texts))
            
            winner, response = judge.compare_pair(query, docs_a, docs_b)
            
            if winner == "A":
                wins[i] += 1
                if verbose:
                    print(f"  {variants[i].variant_id} vs {variants[j].variant_id}: {variants[i].variant_id} wins")
            else:
                wins[j] += 1
                if verbose:
                    print(f"  {variants[i].variant_id} vs {variants[j].variant_id}: {variants[j].variant_id} wins")
    
    # Find the variant with most wins
    best_idx = max(wins.keys(), key=lambda k: wins[k])
    return best_idx, wins


def pairwise_selection_parallel(
    query: str,
    variants: List[VariantResult],
    judge: PairwiseJudge,
    max_workers: int = 10,
) -> Tuple[int, Dict[int, int]]:
    """
    Perform pairwise tournament selection with parallel comparisons.
    """
    n = len(variants)
    wins = {i: 0 for i in range(n)}
    
    # Generate all pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def compare_pair(pair_idx):
        i, j = pair_idx
        docs_a = list(zip(variants[i].doc_ids, variants[i].doc_texts))
        docs_b = list(zip(variants[j].doc_ids, variants[j].doc_texts))
        winner, _ = judge.compare_pair(query, docs_a, docs_b)
        return i, j, winner
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compare_pair, pair): pair for pair in pairs}
        
        for future in as_completed(futures):
            i, j, winner = future.result()
            if winner == "A":
                wins[i] += 1
            else:
                wins[j] += 1
    
    best_idx = max(wins.keys(), key=lambda k: wins[k])
    return best_idx, wins


def main():
    parser = argparse.ArgumentParser(description="Pairwise selection of best query variant")
    parser.add_argument(
        "--run_files", 
        type=str, 
        nargs='+', 
        required=True,
        help="Paths to TREC run files for each variant (space-separated)"
    )
    parser.add_argument(
        "--variant_names",
        type=str,
        nargs='+',
        help="Names for each variant (optional, defaults to run file names)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name for fetching queries (e.g., dl19-passage, dl20-passage)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for judging"
    )
    parser.add_argument(
        "--output_run",
        type=str,
        required=True,
        help="Path to save the output TREC run file with best variants"
    )
    parser.add_argument(
        "--output_stats",
        type=str,
        help="Path to save detailed selection statistics (JSON)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top documents to consider per variant"
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=1234,
        help="Port for vLLM server (for OS LLM)"
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="Host for vLLM server (for OS LLM)"
    )
    parser.add_argument(
        "--use_os_llm",
        action="store_true",
        help="Use open-source LLM via vLLM"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run pairwise comparisons in parallel"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Max workers for parallel execution"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison info"
    )
    
    args = parser.parse_args()
    load_dotenv()
    
    # Validate inputs
    if args.variant_names and len(args.variant_names) != len(args.run_files):
        raise ValueError("Number of variant names must match number of run files")
    
    variant_names = args.variant_names or [os.path.basename(f) for f in args.run_files]
    
    print(f"Loading {len(args.run_files)} variant run files...")
    
    # ---------------------------------------------------------
    # 1. Load all run files
    # ---------------------------------------------------------
    all_runs = {}
    for run_file, name in zip(args.run_files, variant_names):
        print(f"  Loading {name}: {run_file}")
        all_runs[name] = load_trec_run(run_file, top_k=args.top_k)
    
    # Get all unique query IDs across all variants
    all_qids = set()
    for run_data in all_runs.values():
        all_qids.update(run_data.keys())
    all_qids = sorted(all_qids)
    
    print(f"Found {len(all_qids)} unique queries across all variants")
    
    # ---------------------------------------------------------
    # 2. Load query mappings
    # ---------------------------------------------------------
    print(f"Loading queries from {args.dataset_name}...")
    query_mappings = qrel_utils.get_query_mappings(args.dataset_name)
    
    # ---------------------------------------------------------
    # 3. Initialize judge
    # ---------------------------------------------------------
    print(f"Initializing judge with model: {args.model}")
    judge = PairwiseJudge(
        model_name=args.model,
        dataset_name=args.dataset_name,
        vllm_port=args.vllm_port,
        vllm_host=args.vllm_host,
        use_os_llm=args.use_os_llm,
    )
    
    # ---------------------------------------------------------
    # 4. Process each query
    # ---------------------------------------------------------
    results = {}  # qid -> best variant results
    stats = {}    # qid -> selection statistics
    
    print("Processing queries with pairwise selection...")
    for qid in tqdm(all_qids):
        # Get query text
        q_text = None
        if str(qid) in query_mappings:
            q_text = query_mappings[str(qid)]['title']
        elif qid.isdigit() and int(qid) in query_mappings:
            q_text = query_mappings[int(qid)]['title']
        else:
            print(f"Skipping QID {qid}: Not found in query mappings")
            continue
        
        # Build variants for this query
        variants = []
        for name in variant_names:
            if qid not in all_runs[name]:
                continue
            
            doc_ids = [doc_id for doc_id, _ in all_runs[name][qid]]
            scores = [score for _, score in all_runs[name][qid]]
            
            # Fetch document texts
            doc_texts = []
            for doc_id in doc_ids:
                try:
                    text = qrel_utils.get_passage_wrapper(args.dataset_name, doc_id)
                    doc_texts.append(text)
                except Exception as e:
                    print(f"Error fetching doc {doc_id}: {e}")
                    doc_texts.append("")
            
            variants.append(VariantResult(
                variant_id=name,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                scores=scores,
            ))
        
        if len(variants) < 2:
            print(f"Skipping QID {qid}: Less than 2 variants available")
            if len(variants) == 1:
                results[qid] = variants[0]
            continue
        
        # Perform pairwise selection
        if args.verbose:
            print(f"\nQuery {qid}: {q_text}")
        
        if args.parallel:
            best_idx, wins = pairwise_selection_parallel(
                q_text, variants, judge, max_workers=args.max_workers
            )
        else:
            best_idx, wins = pairwise_selection(
                q_text, variants, judge, verbose=args.verbose
            )
        
        best_variant = variants[best_idx]
        results[qid] = best_variant
        
        # Record statistics
        stats[qid] = {
            "query": q_text,
            "winner": best_variant.variant_id,
            "wins": {variants[i].variant_id: wins[i] for i in range(len(variants))},
            "num_variants": len(variants),
        }
        
        if args.verbose:
            print(f"  Winner: {best_variant.variant_id} with {wins[best_idx]} wins")
    
    # ---------------------------------------------------------
    # 5. Write output TREC run file
    # ---------------------------------------------------------
    print(f"Writing output run file to {args.output_run}...")
    with open(args.output_run, 'w') as f:
        for qid in sorted(results.keys()):
            variant = results[qid]
            for rank, (doc_id, score) in enumerate(zip(variant.doc_ids, variant.scores), 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score} pairwise_best\n")
    
    # ---------------------------------------------------------
    # 6. Write statistics (optional)
    # ---------------------------------------------------------
    if args.output_stats:
        print(f"Writing statistics to {args.output_stats}...")
        
        # Aggregate statistics
        variant_wins_total = defaultdict(int)
        for qid_stats in stats.values():
            variant_wins_total[qid_stats["winner"]] += 1
        
        output_stats = {
            "per_query": stats,
            "aggregate": {
                "total_queries": len(stats),
                "variant_selection_counts": dict(variant_wins_total),
            }
        }
        
        with open(args.output_stats, 'w') as f:
            json.dump(output_stats, f, indent=2)
    
    # ---------------------------------------------------------
    # 7. Print summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    variant_wins_total = defaultdict(int)
    for qid_stats in stats.values():
        variant_wins_total[qid_stats["winner"]] += 1
    
    print(f"Total queries processed: {len(stats)}")
    print("\nVariant selection counts:")
    for name, count in sorted(variant_wins_total.items(), key=lambda x: -x[1]):
        pct = count / len(stats) * 100 if stats else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput run file: {args.output_run}")
    if args.output_stats:
        print(f"Statistics file: {args.output_stats}")


if __name__ == "__main__":
    main()
