#!/usr/bin/env python3
"""
End-to-End Pipeline for QueryGym + Pyserini

This script orchestrates the complete pipeline:
1. Load queries from Pyserini topics
2. Reformulate queries using QueryGym
3. Retrieve documents using Pyserini
4. Evaluate results using trec_eval

Supports multiple LLM providers:
- OpenAI (via --api-type openai)
- OpenRouter (via --api-type openrouter)
- Self-hosted vLLM (via --api-type vllm)

Usage:
    # With OpenAI
    python examples/querygym_pyserini/pipeline.py \
        --dataset msmarco-v1-passage.trecdl2019 \
        --method query2doc \
        --model gpt-4 \
        --api-type openai \
        --api-key sk-... \
        --output-dir outputs/dl19_query2doc
    
    # With OpenRouter
    python examples/querygym_pyserini/pipeline.py \
        --dataset msmarco-v1-passage.trecdl2019 \
        --method query2doc \
        --model meta-llama/llama-3.1-8b-instruct \
        --api-type openrouter \
        --api-key sk-or-... \
        --output-dir outputs/dl19_query2doc
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.querygym_pyserini.utils import (
    setup_logging,
    format_time,
    print_dataset_info,
    list_available_datasets,
    save_config
)
from examples.querygym_pyserini import reformulate_queries
from examples.querygym_pyserini import retrieve
from examples.querygym_pyserini import evaluate


def run_pipeline(
    dataset_name: str,
    method: str,
    model: str,
    output_dir: Path,
    steps: List[str],
    llm_config: dict,
    retrieval_config: dict,
    method_params: dict,
    registry_path: str = "dataset_registry.yaml",
    filter_queries_file: str = None,
    num_workers: int = 1
) -> None:
    """
    Run the complete or partial pipeline.
    
    Args:
        dataset_name: Dataset name from registry
        method: QueryGym reformulation method
        model: LLM model name
        output_dir: Output directory
        steps: List of steps to run ['reformulate', 'retrieve', 'evaluate']
        llm_config: LLM configuration
        retrieval_config: Retrieval configuration
        registry_path: Path to dataset registry
    """
    logging.info("="*60)
    logging.info("QueryGym + Pyserini Pipeline")
    logging.info("="*60)
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Method: {method}")
    logging.info(f"Model: {model}")
    if llm_config.get('base_url'):
        api_type_display = "OpenRouter" if "openrouter.ai" in llm_config['base_url'] else \
                          "vLLM" if "localhost" in llm_config['base_url'] or "127.0.0.1" in llm_config['base_url'] else \
                          "OpenAI" if "openai.com" in llm_config['base_url'] else \
                          "Custom"
        logging.info(f"API: {api_type_display} ({llm_config['base_url']})")
    if num_workers > 1:
        logging.info(f"Concurrent workers: {num_workers}")
    logging.info(f"Steps: {', '.join(steps)}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*60)
    
    pipeline_start = time.time()
    results = {}
    
    # Paths for intermediate files
    reformulated_queries_file = output_dir / 'queries' / 'reformulated_queries.tsv'
    run_file = output_dir / 'runs' / 'run.txt'
    
    # Step 1: Reformulate queries
    if 'reformulate' in steps:
        logging.info("\n" + "="*60)
        logging.info("STEP 1: Query Reformulation")
        logging.info("="*60)
        
        step_start = time.time()
        
        try:
            metadata = reformulate_queries.reformulate_queries(
                dataset_name=dataset_name,
                method=method,
                model=model,
                output_dir=output_dir,
                llm_config=llm_config,
                method_params=method_params,
                registry_path=registry_path,
                filter_queries_file=filter_queries_file,
                num_workers=num_workers
            )
            results['reformulation'] = metadata
            results['reformulation']['step_time'] = time.time() - step_start
            
            logging.info(f"✓ Reformulation completed in {format_time(time.time() - step_start)}")
            
        except Exception as e:
            logging.error(f"✗ Reformulation failed: {e}")
            raise
    
    # Step 2: Retrieve documents
    if 'retrieve' in steps:
        logging.info("\n" + "="*60)
        logging.info("STEP 2: Document Retrieval")
        logging.info("="*60)
        
        # Check if queries file exists
        if not reformulated_queries_file.exists():
            logging.error(f"Queries file not found: {reformulated_queries_file}")
            logging.error("Run reformulation first or provide --queries argument")
            sys.exit(1)
        
        step_start = time.time()
        
        try:
            metadata = retrieve.retrieve_documents(
                dataset_name=dataset_name,
                queries_file=reformulated_queries_file,
                output_dir=output_dir,
                k=retrieval_config['k'],
                threads=retrieval_config['threads'],
                registry_path=registry_path
            )
            results['retrieval'] = metadata
            results['retrieval']['step_time'] = time.time() - step_start
            
            logging.info(f"✓ Retrieval completed in {format_time(time.time() - step_start)}")
            
        except Exception as e:
            logging.error(f"✗ Retrieval failed: {e}")
            raise
    
    # Step 3: Evaluate results
    if 'evaluate' in steps:
        logging.info("\n" + "="*60)
        logging.info("STEP 3: Evaluation")
        logging.info("="*60)
        
        # Check if run file exists
        if not run_file.exists():
            logging.error(f"Run file not found: {run_file}")
            logging.error("Run retrieval first")
            sys.exit(1)
        
        step_start = time.time()
        
        try:
            metadata = evaluate.evaluate_run(
                dataset_name=dataset_name,
                run_file=run_file,
                output_dir=output_dir,
                registry_path=registry_path
            )
            results['evaluation'] = metadata
            results['evaluation']['step_time'] = time.time() - step_start
            
            logging.info(f"✓ Evaluation completed in {format_time(time.time() - step_start)}")
            
        except Exception as e:
            logging.error(f"✗ Evaluation failed: {e}")
            raise
    
    # Pipeline complete
    pipeline_time = time.time() - pipeline_start
    
    # Save pipeline summary
    summary = {
        'pipeline': {
            'dataset': dataset_name,
            'method': method,
            'model': model,
            'steps_completed': steps,
            'total_time_seconds': pipeline_time,
            'formatted_time': format_time(pipeline_time)
        },
        'results': results
    }
    
    summary_file = output_dir / 'pipeline_summary.json'
    save_config(summary, summary_file)
    
    # Create human-readable summary
    summary_txt = output_dir / 'pipeline_summary.txt'
    with open(summary_txt, 'w') as f:
        f.write("QueryGym + Pyserini Pipeline Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset:  {dataset_name}\n")
        f.write(f"Method:   {method}\n")
        f.write(f"Model:    {model}\n")
        f.write(f"Steps:    {', '.join(steps)}\n\n")
        f.write(f"Total Pipeline Time: {format_time(pipeline_time)}\n\n")
        
        # Step timings
        f.write("Step Timings:\n")
        f.write("-"*80 + "\n")
        for step, data in results.items():
            if 'step_time' in data:
                f.write(f"  {step:15s}: {format_time(data['step_time'])}\n")
        f.write("\n")
        
        # Evaluation metrics (if available)
        if 'evaluation' in results and 'results' in results['evaluation']:
            f.write("Evaluation Metrics:\n")
            f.write("-"*80 + "\n")
            for metric, value in results['evaluation']['results'].items():
                f.write(f"  {metric:25s}: {value:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(f"Output directory: {output_dir}\n")
    
    logging.info("\n" + "="*60)
    logging.info("Pipeline Summary")
    logging.info("="*60)
    logging.info(f"Total time: {format_time(pipeline_time)}")
    
    if 'evaluation' in results and 'results' in results['evaluation']:
        logging.info("\nKey Metrics:")
        for metric, value in list(results['evaluation']['results'].items())[:5]:
            logging.info(f"  {metric:20s}: {value:.4f}")
    
    logging.info(f"\nOutput: {output_dir}")
    logging.info(f"Summary: {summary_txt}")
    logging.info("="*60)
    logging.info("✓ Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end QueryGym + Pyserini pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (zero-shot) with OpenAI
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model gpt-4 \\
      --api-type openai \\
      --api-key sk-... \\
      --output-dir outputs/dl19_query2doc

  # Full pipeline with OpenRouter
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model meta-llama/llama-3.1-8b-instruct \\
      --api-type openrouter \\
      --api-key sk-or-... \\
      --output-dir outputs/dl19_query2doc_openrouter

  # Full pipeline with self-hosted vLLM
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model your-model-name \\
      --api-type vllm \\
      --base-url http://localhost:8000/v1 \\
      --output-dir outputs/dl19_query2doc_vllm

  # Full pipeline with concurrent processing (8 workers)
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model your-model-name \\
      --num-workers 8 \\
      --output-dir outputs/dl19_query2doc_parallel

  # Query2Doc with few-shot mode
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model your-model-name \\
      --mode fs \\
      --num-examples 4 \\
      --dataset-type msmarco \\
      --collection-path /path/to/collection.tsv \\
      --train-queries-path /path/to/train_queries.tsv \\
      --train-qrels-path /path/to/train_qrels.tsv \\
      --output-dir outputs/dl19_query2doc_fs

  # Only reformulation and retrieval
  python examples/querygym_pyserini/pipeline.py \\
      --dataset beir-v1.0.0-nfcorpus \\
      --method query2doc \\
      --model your-model-name \\
      --steps reformulate,retrieve \\
      --output-dir outputs/nfcorpus_q2d

  # Resume from retrieval (queries already reformulated)
  python examples/querygym_pyserini/pipeline.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model your-model-name \\
      --steps retrieve,evaluate \\
      --output-dir outputs/dl19_query2doc

  # List available datasets
  python examples/querygym_pyserini/pipeline.py --list-datasets
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name from dataset_registry.yaml'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='QueryGym reformulation method'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        type=str,
        help='LLM model name (required)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: outputs/<dataset>_<method>)'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Pipeline steps to run: "all" or comma-separated list of reformulate,retrieve,evaluate (default: all)'
    )
    
    # LLM configuration
    parser.add_argument(
        '--api-type',
        type=str,
        choices=['openai', 'vllm', 'openrouter'],
        help='API type: "openai" (default OpenAI API), "vllm" (self-hosted vLLM), or "openrouter" (OpenRouter API). Automatically sets base_url if not provided.'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        help='LLM API base URL (uses querygym/config/defaults.yaml if not specified). For OpenRouter, defaults to https://openrouter.ai/api/v1'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='LLM API key (uses querygym/config/defaults.yaml if not specified). For OpenRouter, can also use OPENROUTER_API_KEY environment variable'
    )             
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='LLM temperature (default: 1.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='LLM max tokens (default: 128)'
    )
    
    # Retrieval configuration
    parser.add_argument(
        '--k',
        type=int,
        default=1000,
        help='Number of documents to retrieve (default: 1000)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=16,
        help='Number of threads for retrieval (default: 16)'
    )
    
    # Method-specific parameters (for query2doc, query2e, genqr, etc.)
    parser.add_argument(
        '--mode',
        type=str,
        help='Reformulation mode. For query2doc: "zs" (zero-shot), "cot" (chain-of-thought), "fs"/"fewshot" (few-shot). For genqr: "keywords"/"expansion" (default) or "keywords_variant"/"variant".'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=4,
        help='Number of few-shot examples (for few-shot mode, default: 4)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        help='Dataset type for few-shot examples: "msmarco", "beir", or "generic"'
    )
    parser.add_argument(
        '--collection-path',
        type=str,
        help='Path to collection file (for MS MARCO/generic few-shot mode)'
    )
    parser.add_argument(
        '--train-queries-path',
        type=str,
        help='Path to training queries file (for few-shot mode)'
    )
    parser.add_argument(
        '--train-qrels-path',
        type=str,
        help='Path to training qrels file (for few-shot mode)'
    )
    parser.add_argument(
        '--beir-data-dir',
        type=str,
        help='Path to BEIR dataset directory (for BEIR few-shot mode)'
    )
    parser.add_argument(
        '--train-split',
        type=str,
        default='train',
        choices=['train', 'dev'],
        help='BEIR split to use for few-shot examples (default: train)'
    )
    parser.add_argument(
        '--qrel-paths',
        type=str,
        help='Comma-separated paths to judge qrel files (e.g. 0010.qrel) for csqe_modified/lamer_modified.'
    )
    parser.add_argument(
        '--judge-rel-mode',
        type=str,
        choices=['positive', 'negative', 'both'],
        default='positive',
        help='Which qrel rows to use for judge filtering: positive (rel>0), negative (rel==0), or both. Default: positive.'
    )

    # Other options
    parser.add_argument(
        '--registry-path',
        type=str,
        default='dataset_registry.yaml',
        help='Path to dataset registry (default: dataset_registry.yaml)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    parser.add_argument(
        '--dataset-info',
        type=str,
        help='Show info about a specific dataset and exit'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--filter-queries',
        type=str,
        help='Path to TSV file (qid\\tquery_text) to filter queries by qid. Only queries with qids in this file will be processed.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of concurrent workers for reformulation (default: 1, sequential). Set to >1 for parallel processing.'
    )
    
    args = parser.parse_args()
    
    # Handle --list-datasets
    if args.list_datasets:
        datasets = list_available_datasets(args.registry_path)
        print("\nAvailable datasets:")
        print("="*60)
        for ds in datasets:
            print(f"  - {ds}")
        print(f"\nTotal: {len(datasets)} datasets")
        return
    
    # Handle --dataset-info
    if args.dataset_info:
        print_dataset_info(args.dataset_info, args.registry_path)
        return
    
    # Validate required arguments
    if not args.dataset or not args.method or not args.model:
        parser.error("--dataset, --method, and --model are required (unless using --list-datasets or --dataset-info)")
    
    # Parse steps
    if args.steps == 'all':
        steps = ['reformulate', 'retrieve', 'evaluate']
    else:
        steps = [s.strip() for s in args.steps.split(',')]
        valid_steps = {'reformulate', 'retrieve', 'evaluate'}
        invalid = set(steps) - valid_steps
        if invalid:
            parser.error(f"Invalid steps: {invalid}. Valid steps: {valid_steps}")
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/{args.dataset}_{args.method}")
    
    # Setup logging
    setup_logging(
        log_dir=args.output_dir / 'logs',
        log_level=args.log_level,
        log_to_file=True
    )
    
    # Auto-configure base_url based on api_type if not explicitly provided
    if args.api_type:
        if args.api_type == 'openrouter':
            # OpenRouter base URL
            if not args.base_url:
                args.base_url = 'https://openrouter.ai/api/v1'
            # Try to get API key from environment if not provided
            if not args.api_key:
                args.api_key = os.getenv('OPENROUTER_API_KEY')
        elif args.api_type == 'vllm':
            # vLLM typically runs on localhost:8000
            if not args.base_url:
                args.base_url = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')
        elif args.api_type == 'openai':
            # OpenAI uses default, but allow override
            if not args.base_url:
                args.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            # Try to get API key from environment if not provided
            if not args.api_key:
                args.api_key = os.getenv('OPENAI_API_KEY')
    
    # Prepare configurations
    llm_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,

    }
    
    # Only include base_url and api_key if explicitly provided or auto-configured
    if args.base_url:
        llm_config['base_url'] = args.base_url
    if args.api_key:
        llm_config['api_key'] = args.api_key
    
    retrieval_config = {
        'k': args.k,
        'threads': args.threads
    }
    
    # Prepare method parameters
    method_params = {}
    if args.mode:
        method_params['mode'] = args.mode
    if args.num_examples:
        method_params['num_examples'] = args.num_examples
    if args.dataset_type:
        method_params['dataset_type'] = args.dataset_type
    if args.collection_path:
        method_params['collection_path'] = args.collection_path
    if args.train_queries_path:
        method_params['train_queries_path'] = args.train_queries_path
    if args.train_qrels_path:
        method_params['train_qrels_path'] = args.train_qrels_path
    if args.beir_data_dir:
        method_params['beir_data_dir'] = args.beir_data_dir
    if args.train_split:
        method_params['train_split'] = args.train_split
    if args.qrel_paths:
        method_params['qrel_paths'] = [p.strip() for p in args.qrel_paths.split(',') if p.strip()]
    method_params['judge_rel_mode'] = getattr(args, 'judge_rel_mode', 'positive')

    try:
        # Run pipeline
        run_pipeline(
            dataset_name=args.dataset,
            method=args.method,
            model=args.model,
            output_dir=args.output_dir,
            steps=steps,
            llm_config=llm_config,
            retrieval_config=retrieval_config,
            method_params=method_params,
            registry_path=args.registry_path,
            filter_queries_file=args.filter_queries,
            num_workers=args.num_workers
        )
        
    except KeyboardInterrupt:
        logging.warning("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

