"""
LLM judge for (original_query, LLM-generated document) pairs.

Input: src.json from query2doc judge dirs, with structure:
  { "qid": { "original_query": str, "reformulated_queries": [doc1, doc2, ...] } }
  (reformulated_queries are the LLM-generated documents, typically 10 per query)

For each (original_query, generated_doc) pair, runs the same relevance judge (0-3)
and writes scores per qid and doc index.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

from umbrela.gpt_judge import GPTJudge
from umbrela.gemini_judge import GeminiJudge
from umbrela.os_llm_judge import OSLLMJudge
from umbrela.together_llm_judge import TogetherLLMJudge
from umbrela.openrouter_llm_judge import OpenRouterLLMJudge
from umbrela.utils import common_utils


def load_src_json(src_path):
    """Load judge src.json: { qid: { original_query, reformulated_queries } }."""
    with open(src_path, "r") as f:
        data = json.load(f)
    return data


def build_query_document_pairs(src_data, max_doc_len=None):
    """
    Build (query, document) pairs and metadata for judging.
    Returns (query_passage_pairs, metadata) where metadata[i] = (qid, doc_idx).
    Optionally truncate document to max_doc_len chars to control prompt size.
    """
    query_passage_pairs = []
    metadata = []
    for qid, item in src_data.items():
        orig = item["original_query"]
        docs = item.get("reformulated_queries", [])
        for doc_idx, doc in enumerate(docs):
            if max_doc_len and len(doc) > max_doc_len:
                doc = doc[:max_doc_len] + "..."
            query_passage_pairs.append((orig, doc))
            metadata.append((str(qid), doc_idx))
    return query_passage_pairs, metadata


def extract_score(raw_output):
    """Extract relevance score 0-3 from LLM response."""
    if not raw_output:
        return 0
    match = re.search(r"[0-3]", raw_output)
    return int(match.group(0)) if match else 0


def main():
    parser = argparse.ArgumentParser(
        description="LLM judge (original query, generated document) pairs from query2doc src.json"
    )
    parser.add_argument(
        "--input_src",
        type=str,
        required=True,
        help="Path to src.json (e.g. .../judge/2019/query2doc_zs/src.json) or directory containing src.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path: JSON with {qid: [scores]} and optional TSV (qid, doc_idx, score)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for judging",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="bing",
    )
    parser.add_argument(
        "--few_shot_count",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=None,
        help="Max characters per generated document (truncate to control prompt size)",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--use_os_llm",
        action="store_true",
    )
    parser.add_argument(
        "--use_openrouter_llm",
        action="store_true",
    )
    parser.add_argument(
        "--use_together_llm",
        action="store_true",
    )
    parser.add_argument(
        "--use_batch",
        action="store_true",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_save_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--keep_batch_files",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_keep_batch_files",
        action="store_true",
    )
    parser.add_argument(
        "--dummy_qrel",
        type=str,
        default="dl19-passage",
        help="Dummy qrel name for judge init (not used for doc judging)",
    )

    args = parser.parse_args()
    load_dotenv()

    # Resolve input path: if directory, use <dir>/src.json
    input_src = args.input_src
    if os.path.isdir(input_src):
        input_src = os.path.join(input_src, "src.json")
    if not os.path.isfile(input_src):
        raise FileNotFoundError(f"src.json not found: {input_src}")

    # Load data and build pairs
    print(f"Loading {input_src}...")
    src_data = load_src_json(input_src)
    query_passage_pairs, metadata = build_query_document_pairs(
        src_data, max_doc_len=args.max_doc_len
    )
    print(f"Total (query, document) pairs: {len(query_passage_pairs)}")

    # Initialize judge (same as judge_and_score_run)
    if args.use_os_llm:
        judge = OSLLMJudge(
            qrel=args.dummy_qrel,
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            vllm_port=args.vllm_port,
            vllm_host=args.vllm_host,
        )
    elif args.use_openrouter_llm:
        judge = OpenRouterLLMJudge(
            qrel=args.dummy_qrel,
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
        )
    elif "gpt" in args.model.lower():
        keep_files = args.keep_batch_files if not args.no_keep_batch_files else False
        judge = GPTJudge(
            qrel=args.dummy_qrel,
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_save_dir=args.batch_save_dir,
            keep_batch_files=keep_files,
        )
    elif args.use_together_llm:
        keep_files = args.keep_batch_files if not args.no_keep_batch_files else False
        judge = TogetherLLMJudge(
            qrel=args.dummy_qrel,
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_save_dir=args.batch_save_dir,
            keep_batch_files=keep_files,
        )
    else:
        judge = GeminiJudge(
            qrel=args.dummy_qrel,
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
        )
    print(f"Judge: {judge.__class__.__name__}")

    # Build prompts and run judging
    judge.query_passage = query_passage_pairs
    judge.prompts = common_utils.generate_prompts(
        judge.query_passage, judge.prompt_examples, judge._prompt_template
    )

    if isinstance(judge, (TogetherLLMJudge, GPTJudge)) and args.use_batch:
        print(f"Using batch processing for {len(judge.prompts)} prompts...")
        outputs = judge.run_batch_requests(judge.prompts, max_new_tokens=200)
    else:

        def judge_single(idx_prompt):
            idx, prompt = idx_prompt
            if isinstance(judge, OSLLMJudge):
                output = judge.run_os_llm(prompt, max_new_tokens=200)
            elif isinstance(judge, GPTJudge):
                output = judge.run_gpt(prompt, max_new_tokens=200)
            elif isinstance(judge, OpenRouterLLMJudge):
                output = judge.run_openrouter_llm(prompt, max_new_tokens=200)
            elif isinstance(judge, TogetherLLMJudge):
                output = judge.run_together_llm(prompt, max_new_tokens=200)
            else:
                output = judge.run_gemini(prompt, max_new_tokens=200)
            return idx, output

        outputs = [None] * len(judge.prompts)
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {
                executor.submit(judge_single, (i, p)): i
                for i, p in enumerate(judge.prompts)
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, output = future.result()
                outputs[idx] = output

    # Aggregate by qid -> [score0, score1, ...]
    qid_scores = {}
    for i, (qid, doc_idx) in enumerate(metadata):
        raw = outputs[i] or ""
        score = extract_score(raw)
        if qid not in qid_scores:
            qid_scores[qid] = []
        # Ensure list length matches doc index (in case of out-of-order)
        while len(qid_scores[qid]) <= doc_idx:
            qid_scores[qid].append(None)
        qid_scores[qid][doc_idx] = score

    # Fill any missing slots with 0
    for qid in qid_scores:
        for j in range(len(qid_scores[qid])):
            if qid_scores[qid][j] is None:
                qid_scores[qid][j] = 0

    # Save JSON
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qid_scores, f, indent=2)
    print(f"Saved scores to {out_path}")

    # Also save TSV alongside: qid  doc_idx  score
    tsv_path = out_path.rsplit(".", 1)[0] + ".tsv"
    with open(tsv_path, "w") as f:
        f.write("qid\tdoc_idx\tscore\n")
        for qid in sorted(qid_scores.keys(), key=lambda x: (x.isdigit(), x)):
            for doc_idx, score in enumerate(qid_scores[qid]):
                f.write(f"{qid}\t{doc_idx}\t{score}\n")
    print(f"Saved TSV to {tsv_path}")

    # Summary
    all_scores = [s for scores in qid_scores.values() for s in scores]
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print(f"Judged {len(all_scores)} pairs; mean score: {avg:.3f}")


if __name__ == "__main__":
    main()
