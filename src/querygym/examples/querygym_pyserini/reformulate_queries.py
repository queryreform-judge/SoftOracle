#!/usr/bin/env python3
"""
Query Reformulation Script for QueryGym + Pyserini Pipeline

This script loads queries from Pyserini topics, reformulates them using QueryGym,
and saves the reformulated queries for retrieval.

Usage:
    python examples/querygym_pyserini/reformulate_queries.py \
        --dataset msmarco-v1-passage.trecdl2019 \
        --method query2doc \
        --model your-model-name \
        --output-dir outputs/dl19_query2doc
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

import querygym as qg
from examples.querygym_pyserini.utils import (
    get_dataset_config,
    load_pyserini_topics,
    setup_logging,
    create_output_dirs,
    save_config,
    format_time,
    print_dataset_info
)


def _reformulate_concurrent(reformulator, queries: List, num_workers: int, ctx_map: Dict[str, Any] = None):
    """
    Reformulate queries concurrently using ThreadPoolExecutor.

    Args:
        reformulator: Reformulator instance
        queries: List of QueryItem objects
        num_workers: Number of concurrent workers
        ctx_map: Optional pre-built context map (qid -> list of contexts or (docid, content) pairs).
                 If None and method requires context, will be built via retrieve_contexts_batch.

    Returns:
        List of ReformulationResult objects in the same order as input queries
    """
    from querygym import QueryItem, ReformulationResult

    # Use provided ctx_map or build if method requires contexts
    if ctx_map is None:
        requires_context = hasattr(reformulator, 'REQUIRES_CONTEXT') and reformulator.REQUIRES_CONTEXT
        if requires_context:
            retrieval_params = reformulator._get_retrieval_params()
            if retrieval_params:
                logging.info("Retrieving contexts for all queries...")
                ctx_map = reformulator.retrieve_contexts_batch(queries, retrieval_params)
        else:
            ctx_map = None
    
    # Create a list to store results in order
    results = [None] * len(queries)
    results_lock = Lock()
    completed_count = [0]  # Use list to allow modification in nested function
    
    # Progress tracking
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(queries), desc=f"Reformulating with {reformulator.cfg.name}", unit="query")
    except ImportError:
        progress_bar = None
    
    def reformulate_single(query_idx: int, query: QueryItem):
        """Reformulate a single query."""
        try:
            ctx = (ctx_map or {}).get(query.qid) if ctx_map else None
            result = reformulator.reformulate(query, ctx)
            
            # Store result at correct index to preserve order
            with results_lock:
                results[query_idx] = result
                completed_count[0] += 1
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Reformulating with {reformulator.cfg.name} "
                        f"({completed_count[0]}/{len(queries)}, QID: {query.qid})"
                    )
            
            return result
        except Exception as e:
            logging.error(f"Error reformulating query {query.qid}: {e}")
            # Create an error result
            error_result = ReformulationResult(
                qid=query.qid,
                original=query.text,
                reformulated=query.text,  # Fallback to original
                metadata={'error': str(e)}
            )
            with results_lock:
                results[query_idx] = error_result
                completed_count[0] += 1
                if progress_bar:
                    progress_bar.update(1)
            return error_result
    
    # Process queries concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(reformulate_single, idx, query): (idx, query)
            for idx, query in enumerate(queries)
        }
        
        # Wait for all to complete (results are stored in order via index)
        for future in as_completed(futures):
            try:
                future.result()  # Get result (or raise exception if any)
            except Exception as e:
                idx, query = futures[future]
                logging.error(f"Failed to reformulate query {query.qid} (index {idx}): {e}")
    
    if progress_bar:
        progress_bar.close()
    
    # Filter out None results (shouldn't happen, but safety check)
    results = [r for r in results if r is not None]
    
    return results


def reformulate_queries(
    dataset_name: str,
    method: str,
    model: str,
    output_dir: Path,
    llm_config: Dict[str, Any],
    method_params: Dict[str, Any],
    registry_path: str = "dataset_registry.yaml",
    filter_queries_file: str = None,
    num_workers: int = 1
) -> Dict[str, Any]:
    """
    Main reformulation function.
    
    Args:
        dataset_name: Name of dataset from registry
        method: QueryGym reformulation method
        model: LLM model name
        output_dir: Output directory
        llm_config: LLM configuration (temperature, max_tokens, etc.)
        method_params: Method-specific parameters
        registry_path: Path to dataset registry
        filter_queries_file: Optional path to TSV file (qid\\tquery_text) to filter queries by qid
        
    Returns:
        Dictionary containing reformulation results and metadata
    """
    logging.info("="*60)
    logging.info("Starting Query Reformulation")
    logging.info("="*60)
    
    start_time = time.time()
    
    # Get dataset configuration
    logging.info(f"Loading dataset: {dataset_name}")
    dataset_config = get_dataset_config(dataset_name, registry_path)
    
    # Support both 'name' (Pyserini) and 'path' (file) for topics
    if 'path' in dataset_config['topics']:
        topic_name_or_path = dataset_config['topics']['path']
    else:
        topic_name_or_path = dataset_config['topics']['name']
    
    index_name = dataset_config['index']['name']
    bm25_weights = dataset_config.get('bm25_weights', {})
    
    # Load queries from Pyserini topics or file
    logging.info(f"Loading topics: {topic_name_or_path}")
    topics = load_pyserini_topics(topic_name_or_path)
    
    # Load filter qids if filter file is provided
    filter_qids = None
    if filter_queries_file:
        logging.info(f"Loading filter queries from: {filter_queries_file}")
        filter_qids = set()
        try:
            with open(filter_queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        qid = str(parts[0].strip())
                        if qid:
                            filter_qids.add(qid)
            logging.info(f"Loaded {len(filter_qids)} qids from filter file")
        except Exception as e:
            logging.error(f"Error loading filter queries file: {e}")
            raise
    
    # Convert Pyserini topics to QueryGym QueryItem format
    # Filter by qids if filter file is provided
    if filter_qids:
        queries = [
            qg.QueryItem(qid=str(qid), text=topic['title'])
            for qid, topic in topics.items()
            if str(qid) in filter_qids
        ]
        logging.info(f"Filtered to {len(queries)} queries (from {len(topics)} total)")
    else:
        queries = [
            qg.QueryItem(qid=str(qid), text=topic['title'])
            for qid, topic in topics.items()
        ]
    logging.info(f"Loaded {len(queries)} queries")
    
    # Initialize Pyserini searcher for methods that need retrieval context
    searcher = None
    try:
        from pyserini.search.lucene import LuceneSearcher
        
        logging.info(f"Initializing Pyserini searcher: {index_name}")
        
        # Create Pyserini searcher (auto-downloads if prebuilt index)
        pyserini_searcher = LuceneSearcher.from_prebuilt_index(index_name)
        
        # Set BM25 parameters if specified
        k1 = bm25_weights.get('k1')
        b = bm25_weights.get('b')
        
        if k1 is not None and b is not None:
            logging.info(f"Setting BM25 weights: k1={k1}, b={b}")
            pyserini_searcher.set_bm25(k1=k1, b=b)
        else:
            logging.info("Using default BM25 weights")
        
        # Wrap the Pyserini searcher with QueryGym's wrapper
        searcher = qg.wrap_pyserini_searcher(pyserini_searcher, answer_key="contents")
        
        logging.info("✓ Searcher initialized and wrapped successfully")
    except Exception as e:
        logging.warning(f"Could not initialize searcher: {e}")
        logging.warning("Some reformulation methods may not work without a searcher")
    
    # Add searcher to method params if available (for methods that need retrieval)
    if searcher is not None:
        method_params['searcher'] = searcher
        retrieval_k = method_params.get('retrieval_k', 10)
        logging.info(f"Searcher configured with retrieval_k={retrieval_k}")
    
    # Create reformulator
    logging.info(f"Creating reformulator: {method}")
    logging.info(f"Model: {model}")
    logging.info(f"LLM config: {llm_config}")
    logging.info(f"Method params: {list(method_params.keys())}")
    
    reformulator = qg.create_reformulator(
        method_name=method,
        model=model,
        params=method_params,
        llm_config=llm_config
    )
    
    # Build ctx_map once for context-requiring methods (normal retrieval or csqe_modified judge-filtered)
    ctx_map = None
    requires_context = hasattr(reformulator, 'REQUIRES_CONTEXT') and reformulator.REQUIRES_CONTEXT
    if requires_context:
        retrieval_params = reformulator._get_retrieval_params()
        if retrieval_params and searcher is not None:
            if method_params.get('qrel_paths'):
                # Judge-filtered: retrieve with larger k, then keep only judge-relevant docs per query
                from querygym.methods.csqe_modified import load_judge_mapping, build_ctx_map_with_judge
                qrel_paths = method_params['qrel_paths']
                if isinstance(qrel_paths, str):
                    qrel_paths = [qrel_paths]
                logging.info("Loading judge mapping from qrel(s): %s", qrel_paths)
                rel_mode = method_params.get("judge_rel_mode", "positive")
                retrieval_k = method_params.get('retrieval_k', 10)
                if rel_mode == "both":
                    judge_docids_map_pos = load_judge_mapping(qrel_paths, rel_mode="positive")
                    judge_docids_map_neg = load_judge_mapping(qrel_paths, rel_mode="negative")
                    all_qids = set(judge_docids_map_pos) | set(judge_docids_map_neg)
                    judge_docids_map = {
                        qid: judge_docids_map_pos.get(qid, []) + judge_docids_map_neg.get(qid, [])
                        for qid in all_qids
                    }
                    k_per_query = retrieval_k * 2
                else:
                    judge_docids_map = load_judge_mapping(qrel_paths, rel_mode=rel_mode)
                    k_per_query = retrieval_k
                k_retrieve = max(200, k_per_query * 20)
                query_texts = [q.text for q in queries]
                query_ids = [q.qid for q in queries]
                logging.info("Retrieving up to %d docs per query for judge filtering...", k_retrieve)
                batch_results = searcher.batch_search(
                    query_texts,
                    k=k_retrieve,
                    num_threads=retrieval_params.get('threads', 16)
                )
                ctx_map = build_ctx_map_with_judge(
                    query_ids, batch_results, judge_docids_map, k_per_query
                )
                logging.info("Built judge-filtered ctx_map for %d queries", len(ctx_map))
            else:
                logging.info("Retrieving contexts for all queries...")
                ctx_map = reformulator.retrieve_contexts_batch(queries, retrieval_params)

    # Reformulate queries
    logging.info("Reformulating queries...")
    if num_workers > 1:
        logging.info(f"Using {num_workers} concurrent workers for parallel processing")

    reformulation_start = time.time()

    # Use concurrent processing if num_workers > 1
    if num_workers > 1:
        results = _reformulate_concurrent(reformulator, queries, num_workers, ctx_map=ctx_map)
    else:
        results = reformulator.reformulate_batch(queries, ctx_map=ctx_map)
    
    reformulation_time = time.time() - reformulation_start
    
    logging.info(f"Reformulation complete in {format_time(reformulation_time)}")
    logging.info(f"Average time per query: {reformulation_time/len(results):.2f}s")
    if num_workers > 1:
        logging.info(f"Used {num_workers} concurrent workers for parallel processing")
    
    # Create output directories
    dirs = create_output_dirs(output_dir)
    
    # Save reformulated queries (concat format for retrieval)
    output_queries_concat = dirs['queries'] / 'reformulated_queries.tsv'
    qg.DataLoader.save_queries(
        [qg.QueryItem(r.qid, r.reformulated) for r in results],
        output_queries_concat,
        format='tsv'
    )
    logging.info(f"Saved reformulated queries (concat): {output_queries_concat}")
    
    # Save original queries for reference
    output_queries_original = dirs['queries'] / 'original_queries.tsv'
    qg.DataLoader.save_queries(queries, output_queries_original, format='tsv')
    logging.info(f"Saved original queries: {output_queries_original}")
    
    # Prepare method params for metadata (exclude non-serializable searcher object)
    metadata_params = {k: v for k, v in method_params.items() if k != 'searcher'}
    
    # Save metadata
    metadata = {
        'dataset': {
            'name': dataset_name,
            'full_name': dataset_config.get('name', ''),
            'topics': topic_name_or_path,
            'index': index_name,
            'num_queries': len(queries),
            'bm25_weights': bm25_weights
        },
        'reformulation': {
            'method': method,
            'model': model,
            'llm_config': llm_config,
            'method_params': metadata_params,
            'searcher': searcher.get_searcher_info() if searcher else None
        },
        'timing': {
            'total_time_seconds': reformulation_time,
            'avg_time_per_query_seconds': reformulation_time / len(results),
            'formatted_time': format_time(reformulation_time)
        },
        'outputs': {
            'reformulated_queries': str(output_queries_concat),
            'original_queries': str(output_queries_original)
        }
    }
    
    metadata_file = dirs['base'] / 'reformulation_metadata.json'
    save_config(metadata, metadata_file)
    
    # Save sample reformulations for inspection
    sample_file = dirs['base'] / 'reformulation_samples.txt'
    with open(sample_file, 'w') as f:
        f.write("Sample Reformulations\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results[:10]):  # First 10
            f.write(f"Query {i+1} (QID: {result.qid}):\n")
            f.write(f"Original:     {result.original}\n")
            f.write(f"Reformulated: {result.reformulated[:200]}...\n")
            if result.metadata:
                f.write(f"Metadata:     {result.metadata}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    logging.info(f"Saved sample reformulations: {sample_file}")
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"Reformulation completed in {format_time(total_time)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("="*60)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Reformulate queries using QueryGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python examples/querygym_pyserini/reformulate_queries.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --method query2doc \\
      --model your-model-name

  # With custom LLM config
  python examples/querygym_pyserini/reformulate_queries.py \\
      --dataset beir-v1.0.0-nfcorpus \\
      --method query2doc \\
      --model your-model-name \\
      --temperature 0.7 \\
      --max-tokens 256

  # List available datasets
  python examples/querygym_pyserini/reformulate_queries.py --list-datasets
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name from dataset_registry.yaml'
    )
    parser.add_argument(
        '--method',
        type=str,
        help='QueryGym reformulation method (genqr, genqr_ensemble, query2doc, etc.)'
    )
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
        '--base-url',
        type=str,
        help='LLM API base URL (uses querygym/config/defaults.yaml if not specified)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='LLM API key (uses querygym/config/defaults.yaml if not specified)'
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
    parser.add_argument(
        '--retrieval-k',
        type=int,
        default=10,
        help='Number of documents to retrieve for methods that need context (default: 10)'
    )
    parser.add_argument(
        '--qrel-paths',
        type=str,
        help='Comma-separated paths to judge qrel files for csqe_modified/lamer_modified.'
    )
    parser.add_argument(
        '--judge-rel-mode',
        type=str,
        choices=['positive', 'negative', 'both'],
        default='positive',
        help='Which qrel rows to use: positive (rel>0), negative (rel==0), or both. Default: positive.'
    )
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
    
    args = parser.parse_args()
    
    # Handle --list-datasets
    if args.list_datasets:
        from examples.querygym_pyserini.utils import list_available_datasets
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
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/{args.dataset}_{args.method}")
    
    # Setup logging
    setup_logging(
        log_dir=args.output_dir / 'logs',
        log_level=args.log_level,
        log_to_file=True
    )
    
    # Log configuration
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Method: {args.method}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output: {args.output_dir}")
    
    # Prepare configurations
    llm_config = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens
    }
    
    # Only include base_url and api_key if explicitly provided
    if args.base_url:
        llm_config['base_url'] = args.base_url
    if args.api_key:
        llm_config['api_key'] = args.api_key
    
    # Set method parameters
    method_params = {
        'retrieval_k': args.retrieval_k  # Number of docs to retrieve for context
    }
    if args.qrel_paths:
        method_params['qrel_paths'] = [p.strip() for p in args.qrel_paths.split(',') if p.strip()]
    method_params['judge_rel_mode'] = getattr(args, 'judge_rel_mode', 'positive')

    try:
        # Run reformulation
        metadata = reformulate_queries(
            dataset_name=args.dataset,
            method=args.method,
            model=args.model,
            output_dir=args.output_dir,
            llm_config=llm_config,
            method_params=method_params,
            registry_path=args.registry_path,
            filter_queries_file=args.filter_queries
        )
        
        logging.info("✓ Reformulation completed successfully!")
        
    except Exception as e:
        logging.error(f"✗ Reformulation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

