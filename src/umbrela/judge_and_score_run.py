import argparse
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv

# Import your existing utilities
from umbrela.gpt_judge import GPTJudge 
from umbrela.gemini_judge import GeminiJudge
from umbrela.os_llm_judge import OSLLMJudge
from umbrela.together_llm_judge import TogetherLLMJudge
from umbrela.openrouter_llm_judge import OpenRouterLLMJudge
from umbrela.utils import common_utils, qrel_utils
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_trec_run(run_file_path, top_k=10):
    """
    Reads a TREC run file and returns a dict: {qid: [docid1, docid2, ...]}
    Limit to top_k docs per query to save costs.
    """
    run_data = {}
    with open(run_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            # Standard TREC format: qid Q0 docid rank score runtag
            qid = parts[0]
            docid = parts[2]
            rank = int(parts[3])
            

            if qid not in run_data:
                run_data[qid] = []
            run_data[qid].append(docid)
    # only get top_k docs per query
    for qid in run_data:
        run_data[qid] = run_data[qid][:top_k]
    return run_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", type=str, required=True, help="Path to your reformulated TREC run file")
    parser.add_argument("--dataset_name", type=str, required=True, help="e.g., dl19-passage, dl20-passage (to fetch ORIGINAL queries)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name for judging")
    parser.add_argument("--output_qrel", type=str, required=True, help="Path to save the generated QREL file")
    parser.add_argument("--top_k", type=int, default=1000, help="Depth to judge (usually 10 for NDCG@10)")
    parser.add_argument("--prompt_type", type=str, default="bing")
    parser.add_argument("--few_shot_count", type=int, default=0)
    parser.add_argument("--vllm_port", type=int, default=1234, help="Port for vLLM server (for OS LLM)")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="Host for vLLM server (for OS LLM)")
    parser.add_argument("--use_os_llm", action="store_true", help="Use open-source LLM via vLLM")
    parser.add_argument("--use_openrouter_llm", action="store_true", help="Use OpenRouter LLM")
    parser.add_argument("--use_together_llm", action="store_true", help="Use Together LLM")
    parser.add_argument("--use_batch", action="store_true", help="Use batch processing for TogetherLLMJudge or GPTJudge")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for TogetherLLMJudge or GPTJudge")
    parser.add_argument("--batch_save_dir", type=str, default=None, help="Directory to save batch files")
    parser.add_argument("--keep_batch_files", action="store_true", default=True, help="Keep batch files after processing")
    parser.add_argument("--no_keep_batch_files", action="store_true", help="Delete batch files after processing")
    parser.add_argument(
        '--filter-queries',
        type=str,
        help='Path to TSV file (qid\\tquery_text) to filter queries by qid. Only queries with qids in this file will be processed.'
    )
    
    # Dummy qrel just to satisfy class init
    parser.add_argument("--dummy_qrel", type=str, default="dl19-passage")

    args = parser.parse_args()
    load_dotenv()

    # ---------------------------------------------------------
    # 1. Initialize Judge
    # ---------------------------------------------------------
    if args.use_os_llm:
        judge = OSLLMJudge(
            qrel=args.dataset_name, 
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            vllm_port=args.vllm_port,
            vllm_host=args.vllm_host
        )
    elif args.use_openrouter_llm:
        judge = OpenRouterLLMJudge(
            qrel=args.dataset_name, 
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count
        )
    elif 'gpt' in args.model.lower():
        keep_files = args.keep_batch_files if not args.no_keep_batch_files else False
        judge = GPTJudge(
            qrel=args.dataset_name, 
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_save_dir=args.batch_save_dir,
            keep_batch_files=keep_files
        )
    elif args.use_together_llm:
        keep_files = args.keep_batch_files if not args.no_keep_batch_files else False
        judge = TogetherLLMJudge(
            qrel=args.dataset_name, 
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_save_dir=args.batch_save_dir,
            keep_batch_files=keep_files
        )
    else:
        judge = GeminiJudge(
            qrel=args.dataset_name, 
            model_name=args.model,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count
        )
    print(f"Judge: {judge}")
    # ---------------------------------------------------------
    # 2. Prepare Data (Reformulated Docs vs Original Queries)
    # ---------------------------------------------------------
    print(f"Loading original queries from {args.dataset_name}...")
    query_mappings = qrel_utils.get_query_mappings(args.dataset_name)

    print(f"Loading run file: {args.run_file}")
    run_data = load_trec_run(args.run_file, top_k=args.top_k)

    # Filter queries if filter file is provided
    if args.filter_queries:
        print(f"Loading filter queries from {args.filter_queries}...")
        filter_qids = set()
        with open(args.filter_queries, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    qid = parts[0].strip()
                    filter_qids.add(qid)
                    # Also add as int if it's numeric, to handle both string and int qids
                    try:
                        filter_qids.add(int(qid))
                    except ValueError:
                        pass
        
        # Filter run_data to only include queries in the filter set
        original_count = len(run_data)
        run_data = {qid: doc_list for qid, doc_list in run_data.items() 
                    if str(qid) in filter_qids or qid in filter_qids}
        filtered_count = len(run_data)
        print(f"Filtered queries: {original_count} -> {filtered_count} (kept {filtered_count} queries)")

    query_passage_pairs = []
    metadata = [] 
    
    

    print("Fetching passages and preparing prompts...")
    for qid, doc_list in tqdm(run_data.items()):
        # Handle QID types (int vs str) common in pyserini
        q_text = None
        if str(qid) in query_mappings:
            q_text = query_mappings[str(qid)]['title']
        elif int(qid) in query_mappings:
            q_text = query_mappings[int(qid)]['title']
        else:
            print(f"Skipping QID {qid}: Not found in original dataset.")
            continue

        for docid in doc_list:
            try:
                # Fetch passage text
                passage_text = qrel_utils.get_passage_wrapper(args.dataset_name, docid)
                query_passage_pairs.append((q_text, passage_text))
                metadata.append((qid, docid))
            except Exception as e:
                print(f"Error fetching doc {docid}: {e}")
    # ---------------------------------------------------------
    # 3. Generate Judgments
    # ---------------------------------------------------------
    print(f"Judging {len(query_passage_pairs)} pairs...")
    
    # Manually inject data into judge instance
    judge.query_passage = query_passage_pairs
    judge.prompts = common_utils.generate_prompts(
        judge.query_passage, judge.prompt_examples, judge._prompt_template
    )
    
    # Use batch processing for TogetherLLMJudge or GPTJudge if enabled 
    if isinstance(judge, (TogetherLLMJudge, GPTJudge)) and args.use_batch:
        print(f"Using batch processing for {len(judge.prompts)} prompts...")
        outputs = judge.run_batch_requests(judge.prompts, max_new_tokens=200)
    else:
        # Use individual requests with ThreadPoolExecutor
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
            else:  # GeminiJudge
                output = judge.run_gemini(prompt, max_new_tokens=200)
            return idx, output

        outputs = [None] * len(judge.prompts)

        with ThreadPoolExecutor(max_workers=200) as executor:  # Adjust max_workers as needed
            futures = {executor.submit(judge_single, (i, p)): i for i, p in enumerate(judge.prompts)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, output = future.result()
                outputs[idx] = output


    # outputs = []
    # for prompt in tqdm(judge.prompts):
    #     if 'gpt' in args.model.lower(): 
    #         output = judge.run_gpt(prompt, max_new_tokens=200)
    #     else:
    #         output = judge.run_gemini(prompt, max_new_tokens=200)

    #     outputs.append(output)

    # ---------------------------------------------------------
    # 4. Save QREL File
    # ---------------------------------------------------------
    print(f"Saving generated labels to {args.output_qrel}...")
    
    with open(args.output_qrel, 'w') as f_out:
        for i, raw_output in enumerate(outputs):
            qid, docid = metadata[i]
            
            # Extract score (0-3) from LLM response
            # This regex looks for the first digit in the response
            match = re.search(r'\d', raw_output)
            score = match.group(0) if match else "0"
            
            # Write standard QREL format: qid Q0 docid score
            f_out.write(f"{qid} Q0 {docid} {score}\n")

    # ---------------------------------------------------------
    # 5. Calculate NDCG@10
    # ---------------------------------------------------------
    print("-" * 50)
    print("Calculating NDCG@10 score...")
    print("-" * 50)
    
    try:
        # We use the utility function you already have in qrel_utils
        ndcg_score = qrel_utils.fetch_ndcg_score(args.output_qrel, args.run_file)
        print(f"\nFinal NDCG@10 Score: {ndcg_score}")
    except Exception as e:
        print(f"Error calculating NDCG: {e}")
        print("Ensure pyserini is installed and fetch_ndcg_score is accessible.")

if __name__ == "__main__":
    main()