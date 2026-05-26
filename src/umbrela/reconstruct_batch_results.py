#!/usr/bin/env python3
"""
Helper script to reconstruct batch results from chunk mapping files.
This makes it easy to map results back to original request indices.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple


def load_chunk_mapping(mapping_file: str) -> Dict:
    """Load chunk mapping file."""
    with open(mapping_file, "r") as f:
        return json.load(f)


def reconstruct_results(mapping_file: str, output_file: str = None) -> List[str]:
    """
    Reconstruct full results list from chunk mapping file.
    
    Args:
        mapping_file: Path to chunk_mapping_{timestamp}.json file
        output_file: Optional path to save reconstructed results
        
    Returns:
        List of outputs in original order
    """
    mapping_data = load_chunk_mapping(mapping_file)
    
    total_requests = mapping_data["total_requests"]
    chunks = sorted(mapping_data["chunks"], key=lambda x: x["chunk_num"])
    
    # Initialize results list
    results = [None] * total_requests
    
    print(f"Reconstructing {total_requests} results from {len(chunks)} chunks...")
    
    for chunk_info in chunks:
        chunk_num = chunk_info["chunk_num"]
        request_range = chunk_info["request_range"]
        start_idx = request_range["start"]
        end_idx = request_range["end"]
        
        # Try to load from results_mapping_file first (most detailed)
        if chunk_info.get("results_mapping_file") and os.path.exists(chunk_info["results_mapping_file"]):
            print(f"  Loading chunk {chunk_num} from results mapping file...")
            with open(chunk_info["results_mapping_file"], "r") as f:
                chunk_results = json.load(f)
            
            for result_item in chunk_results["results"]:
                orig_idx = result_item["original_index"]
                if orig_idx < total_requests:
                    results[orig_idx] = result_item["output"]
        
        # Fallback to output file
        elif chunk_info.get("output_file") and os.path.exists(chunk_info["output_file"]):
            print(f"  Loading chunk {chunk_num} from output file...")
            chunk_results = []
            with open(chunk_info["output_file"], "r") as f:
                for line in f:
                    result = json.loads(line.strip())
                    custom_id = result.get("custom_id", "")
                    # Extract index from custom_id: chunk{N}_req{M}
                    if f"chunk{chunk_num}_req" in custom_id:
                        try:
                            req_idx = int(custom_id.split("req")[1])
                            response_body = result.get("response", {}).get("body", {})
                            if isinstance(response_body, dict):
                                choices = response_body.get("choices", [])
                                if choices and len(choices) > 0:
                                    content = choices[0].get("message", {}).get("content", "")
                                    if req_idx < total_requests:
                                        results[req_idx] = content.lower() if content else ""
                        except (ValueError, IndexError):
                            continue
        
        # Check for missing results
        missing = [i for i in range(start_idx, end_idx + 1) if results[i] is None]
        if missing:
            print(f"  Warning: Missing results for indices: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    # Fill None values with empty strings
    results = [r if r is not None else "" for r in results]
    
    if output_file:
        with open(output_file, "w") as f:
            for idx, result in enumerate(results):
                f.write(f"{idx}\t{result}\n")
        print(f"\nReconstructed results saved to: {output_file}")
    
    return results


def print_chunk_summary(mapping_file: str):
    """Print a summary of chunks."""
    mapping_data = load_chunk_mapping(mapping_file)
    
    print(f"\nChunk Mapping Summary")
    print(f"=" * 60)
    print(f"Total Requests: {mapping_data['total_requests']}")
    print(f"Total Chunks: {mapping_data['total_chunks']}")
    print(f"Max Batch Size: {mapping_data['max_batch_size']}")
    print(f"Created At: {mapping_data['created_at']}")
    print(f"\nChunks:")
    print(f"-" * 60)
    
    for chunk in sorted(mapping_data["chunks"], key=lambda x: x["chunk_num"]):
        print(f"Chunk {chunk['chunk_num']}:")
        print(f"  Request Range: {chunk['request_range']['start']} - {chunk['request_range']['end']} ({chunk['request_range']['count']} requests)")
        print(f"  Batch ID: {chunk.get('batch_id', 'N/A')}")
        print(f"  Status: {chunk.get('status', 'N/A')}")
        print(f"  Input File: {os.path.basename(chunk.get('input_file', 'N/A'))}")
        print(f"  Output File: {os.path.basename(chunk.get('output_file', 'N/A'))}")
        if chunk.get('error_file'):
            print(f"  Error File: {os.path.basename(chunk['error_file'])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct batch results from chunk mapping files"
    )
    parser.add_argument(
        "mapping_file",
        type=str,
        help="Path to chunk_mapping_{timestamp}.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save reconstructed results (default: print to stdout)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print chunk summary instead of reconstructing"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return
    
    if args.summary:
        print_chunk_summary(args.mapping_file)
    else:
        results = reconstruct_results(args.mapping_file, args.output)
        if not args.output:
            print("\nReconstructed Results:")
            print("=" * 60)
            for idx, result in enumerate(results[:10]):  # Show first 10
                print(f"{idx}: {result[:100]}...")
            if len(results) > 10:
                print(f"... and {len(results) - 10} more results")


if __name__ == "__main__":
    main()
