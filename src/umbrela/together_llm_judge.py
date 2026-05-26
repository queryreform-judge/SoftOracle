import argparse
import datetime
import json
import os
import tempfile
import time
from typing_extensions import Optional
import random

from dotenv import load_dotenv
from together import Together
from retry import retry
from tqdm import tqdm

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class TogetherLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str = "deepseek-ai/DeepSeek-V3",
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 0,
        use_batch: bool = False,
        batch_size: int = 1000,
        batch_save_dir: Optional[str] = None,
        keep_batch_files: bool = True,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.keep_batch_files = keep_batch_files
        
        # Set up batch file save directory
        if batch_save_dir:
            self.batch_save_dir = batch_save_dir
        else:
            # Default: save in batch_files directory with model name
            model_safe_name = model_name.replace("/", "_").replace("-", "_")
            self.batch_save_dir = f"batch_files/{model_safe_name}"
        
        # Create batch save directory if it doesn't exist
        os.makedirs(self.batch_save_dir, exist_ok=True)
        
        self.create_together_client()

    def create_together_client(self):
        """Initialize Together API client."""
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
        self.client = Together(api_key=api_key)
        self.model = self.model_name

    @retry(tries=3, delay=0.1)
    def run_together_llm(self, prompt, max_new_tokens):
        """Run a single request using Together API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                top_p=1,
            )
            output = (
                response.choices[0].message.content.lower()
                if response.choices[0].message.content
                else ""
            )
            print(f'output: {output}')
        except Exception as e:
            print(f"Encountered {e} for {prompt}")
            output = ""
        return output

    def run_batch_requests(self, prompts, max_new_tokens):
        """Run batch requests using Together Batch API."""
        print(f"Processing {len(prompts)} requests in batch mode...")
        
        # Together Batch API supports up to 50,000 requests per batch
        # Process in chunks if needed (use Together's limit, but respect user's batch_size setting)
        # max_batch_size = min(self.batch_size, 50000)  # Together's max is 50,000
        max_batch_size = 50000

        if len(prompts) > max_batch_size:
            print(f"Warning: {len(prompts)} requests exceed max batch size ({max_batch_size}). Processing in chunks...")
            
            # Create chunk mapping file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_mapping_file = os.path.join(self.batch_save_dir, f"chunk_mapping_{timestamp}.json")
            chunk_mappings = []
            
            all_outputs = []
            chunk_num = 0
            for i in range(0, len(prompts), max_batch_size):
                chunk = prompts[i:i + max_batch_size]
                chunk_end = min(i + max_batch_size, len(prompts))
                chunk_range = (i, chunk_end - 1)  # inclusive range
                
                print(f"Processing chunk {chunk_num}: requests {i} to {chunk_end - 1} (total: {len(chunk)} requests)")
                
                chunk_outputs, chunk_metadata = self._process_batch_chunk(
                    chunk, max_new_tokens, i, chunk_num, chunk_range, timestamp
                )
                
                # Store chunk mapping
                chunk_mappings.append({
                    "chunk_num": chunk_num,
                    "request_range": {"start": i, "end": chunk_end - 1, "count": len(chunk)},
                    "batch_id": chunk_metadata.get("batch_id"),
                    "input_file": chunk_metadata.get("input_file"),
                    "output_file": chunk_metadata.get("output_file"),
                    "error_file": chunk_metadata.get("error_file"),
                    "metadata_file": chunk_metadata.get("metadata_file"),
                    "results_mapping_file": chunk_metadata.get("results_mapping_file"),
                    "status": chunk_metadata.get("status")
                })
                
                all_outputs.extend(chunk_outputs)
                chunk_num += 1
            
            # Save chunk mapping file
            mapping_data = {
                "total_requests": len(prompts),
                "total_chunks": len(chunk_mappings),
                "max_batch_size": max_batch_size,
                "created_at": timestamp,
                "chunks": chunk_mappings
            }
            with open(chunk_mapping_file, "w") as f:
                json.dump(mapping_data, f, indent=2)
            print(f"\nChunk mapping saved to: {chunk_mapping_file}")
            print(f"This file maps all chunks and can be used to reconstruct results later.")
            
            return all_outputs
        else:
            chunk_outputs, chunk_metadata = self._process_batch_chunk(
                prompts, max_new_tokens, 0, 0, (0, len(prompts) - 1), 
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            return chunk_outputs
    
    def _process_batch_chunk(self, prompts, max_new_tokens, offset=0, chunk_num=0, chunk_range=(0, 0), timestamp=None):
        """Process a single batch chunk.
        
        Returns:
            tuple: (outputs list, metadata dict)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare batch input file
        batch_input_lines = []
        for idx, prompt in enumerate(prompts):
            # Use more descriptive custom_id with chunk info
            custom_id = f"chunk{chunk_num}_req{offset + idx}"
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "top_p": 1,
            }
            batch_input_lines.append({
                "custom_id": custom_id,
                "body": body
            })
        
        # Create meaningful filename with request range
        range_str = f"req{chunk_range[0]}-{chunk_range[1]}"
        chunk_suffix = f"_chunk{chunk_num}_{range_str}"
        
        # Write batch input file with meaningful name
        batch_input_filename = f"batch_input_{timestamp}{chunk_suffix}.jsonl"
        batch_input_file = os.path.join(self.batch_save_dir, batch_input_filename)
        
        with open(batch_input_file, "w") as f:
            for line in batch_input_lines:
                f.write(json.dumps(line) + "\n")
        
        print(f"Batch input file saved to: {batch_input_file}")
        
        try:
            # Upload batch input file
            print("Uploading batch input file...")
            file_resp = self.client.files.upload(
                file=batch_input_file,
                purpose="batch-api",
                check=False
            )
            file_id = file_resp.id
            print(f"File uploaded with ID: {file_id}")
            
            # Create batch
            print("Creating batch...")
            batch = self.client.batches.create_batch(
                file_id=file_id,
                endpoint="/v1/chat/completions"
            )
            batch_id = batch.id
            print(f"Batch created with ID: {batch_id}")
            
            # Save batch metadata with chunk information
            metadata_file = os.path.join(self.batch_save_dir, f"batch_metadata_{batch_id}.json")
            metadata = {
                "batch_id": batch_id,
                "file_id": file_id,
                "input_file": batch_input_file,
                "model": self.model,
                "num_requests": len(prompts),
                "chunk_num": chunk_num,
                "request_range": {"start": chunk_range[0], "end": chunk_range[1], "count": len(prompts)},
                "offset": offset,
                "created_at": timestamp,
                "status": "IN_PROGRESS"
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Batch metadata saved to: {metadata_file}")
            
            # Poll for completion
            print("Waiting for batch to complete...")
            max_wait_time = 86400  # 24 hours
            poll_interval = random.randint(30, 60)  # Check every minute
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                batch_status = self.client.batches.get_batch(batch_id)
                status = batch_status.status
                print(f"Batch status: {status}")
                
                if status == "COMPLETED":
                    # Download results with meaningful filename
                    print("Batch completed. Downloading results...")
                    output_filename = f"batch_output_{batch_id}_{timestamp}{chunk_suffix}.jsonl"
                    output_file = os.path.join(self.batch_save_dir, output_filename)
                    
                    self.client.files.retrieve_content(
                        id=batch_status.output_file_id,
                        output=output_file,
                    )
                    print(f"Batch output file saved to: {output_file}")
                    
                    # Parse results
                    outputs = []
                    result_map = {}
                    with open(output_file, "r") as f:
                        for line in f:
                            result = json.loads(line.strip())
                            custom_id = result.get("custom_id")
                            if custom_id:
                                result_map[custom_id] = result
                    
                    # Map results back to original order using chunk-aware custom_id
                    for idx in range(len(prompts)):
                        custom_id = f"chunk{chunk_num}_req{offset + idx}"
                        if custom_id in result_map:
                            result = result_map[custom_id]
                            # Parse response body - the structure may vary
                            response_body = result.get("response", {}).get("body", {})
                            if isinstance(response_body, dict):
                                choices = response_body.get("choices", [])
                                if choices and len(choices) > 0:
                                    content = choices[0].get("message", {}).get("content", "")
                                    outputs.append(content.lower() if content else "")
                                else:
                                    outputs.append("")
                            else:
                                outputs.append("")
                        else:
                            outputs.append("")
                    
                    # Create a results mapping file for this chunk
                    results_mapping_file = os.path.join(
                        self.batch_save_dir, 
                        f"results_mapping_{batch_id}_{timestamp}{chunk_suffix}.json"
                    )
                    results_mapping = {
                        "batch_id": batch_id,
                        "chunk_num": chunk_num,
                        "request_range": {"start": chunk_range[0], "end": chunk_range[1]},
                        "output_file": output_file,
                        "num_results": len(outputs),
                        "results": [
                            {
                                "original_index": offset + idx,
                                "chunk_index": idx,
                                "custom_id": f"chunk{chunk_num}_req{offset + idx}",
                                "output": output
                            }
                            for idx, output in enumerate(outputs)
                        ]
                    }
                    with open(results_mapping_file, "w") as f:
                        json.dump(results_mapping, f, indent=2)
                    print(f"Results mapping saved to: {results_mapping_file}")
                    
                    # Check for errors in error file if available
                    error_file = None
                    if hasattr(batch_status, "error_file_id") and batch_status.error_file_id:
                        error_filename = f"batch_errors_{batch_id}_{timestamp}{chunk_suffix}.jsonl"
                        error_file = os.path.join(self.batch_save_dir, error_filename)
                        try:
                            self.client.files.retrieve_content(
                                id=batch_status.error_file_id,
                                output=error_file,
                            )
                            print(f"Warning: Some requests failed. Error file saved to: {error_file}")
                        except Exception as e:
                            print(f"Could not retrieve error file: {e}")
                            error_file = None
                    
                    # Update metadata with completion status
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        metadata.update({
                            "status": "COMPLETED",
                            "completed_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "output_file": output_file,
                            "error_file": error_file if error_file else None,
                            "results_mapping_file": results_mapping_file,
                            "output_file_id": batch_status.output_file_id,
                            "error_file_id": getattr(batch_status, "error_file_id", None)
                        })
                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2)
                    
                    # Prepare return metadata
                    chunk_metadata = {
                        "batch_id": batch_id,
                        "chunk_num": chunk_num,
                        "request_range": chunk_range,
                        "input_file": batch_input_file,
                        "output_file": output_file,
                        "error_file": error_file,
                        "metadata_file": metadata_file,
                        "results_mapping_file": results_mapping_file,
                        "status": "COMPLETED"
                    }
                    
                    # Cleanup files if not keeping them
                    if not self.keep_batch_files:
                        if os.path.exists(batch_input_file):
                            os.remove(batch_input_file)
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        if error_file and os.path.exists(error_file):
                            os.remove(error_file)
                        if os.path.exists(metadata_file):
                            os.remove(metadata_file)
                        if os.path.exists(results_mapping_file):
                            os.remove(results_mapping_file)
                    
                    return outputs, chunk_metadata
                
                elif status == "FAILED":
                    # Update metadata with failure status
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        metadata["status"] = "FAILED"
                        metadata["failed_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2)
                    chunk_metadata = {
                        "batch_id": batch_id,
                        "chunk_num": chunk_num,
                        "request_range": chunk_range,
                        "input_file": batch_input_file,
                        "metadata_file": metadata_file,
                        "status": "FAILED"
                    }
                    raise Exception(f"Batch processing failed: {batch_id}")
                
                elif status == "CANCELLED":
                    # Update metadata with cancelled status
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        metadata["status"] = "CANCELLED"
                        metadata["cancelled_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(metadata_file, "w") as f:
                            json.dump(metadata, f, indent=2)
                    chunk_metadata = {
                        "batch_id": batch_id,
                        "chunk_num": chunk_num,
                        "request_range": chunk_range,
                        "input_file": batch_input_file,
                        "metadata_file": metadata_file,
                        "status": "CANCELLED"
                    }
                    raise Exception(f"Batch was cancelled: {batch_id}")
                
                # Wait before next poll
                time.sleep(poll_interval)
            
            raise Exception(f"Batch did not complete within {max_wait_time} seconds")
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Update metadata with error status if it exists
            import glob
            if 'batch_id' in locals():
                metadata_pattern = os.path.join(self.batch_save_dir, f"batch_metadata_{batch_id}.json")
                if os.path.exists(metadata_pattern):
                    with open(metadata_pattern, "r") as f:
                        metadata = json.load(f)
                    metadata["status"] = "ERROR"
                    metadata["error"] = str(e)
                    metadata["error_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(metadata_pattern, "w") as f:
                        json.dump(metadata, f, indent=2)
            
            # Fallback to individual requests
            print("Falling back to individual requests...")
            return [self.run_together_llm(prompt, max_new_tokens) for prompt in prompts]

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
    ):
        """Predict using LLM with optional batch processing."""
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )

        if self.use_batch and len(self.prompts) >= self.batch_size:
            # Use batch API for large requests
            outputs = self.run_batch_requests(self.prompts, max_new_tokens)
        else:
            # Use individual requests
            outputs = [
                self.run_together_llm(prompt, max_new_tokens) 
                for prompt in tqdm(self.prompts)
            ]
        return outputs

    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        """Judge requests and return judgments."""
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file", required=True)
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument(
        "--prompt_type", type=str, help="Prompt type. Supported types: [bing, basic, binary]."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-V3",
        help="Together model name (default: deepseek-ai/DeepSeek-V3)"
    )
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument(
        "--use_batch",
        action="store_true",
        help="Use Together Batch API for processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Minimum number of requests to use batch API (default: 1000)"
    )
    parser.add_argument(
        "--batch_save_dir",
        type=str,
        default=None,
        help="Directory to save batch files (default: batch_files/{model_name})"
    )
    parser.add_argument(
        "--keep_batch_files",
        action="store_true",
        help="Keep batch input/output files after processing (default: True)"
    )
    parser.add_argument(
        "--no_keep_batch_files",
        action="store_true",
        help="Delete batch files after processing"
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    keep_files = args.keep_batch_files if not args.no_keep_batch_files else False
    
    judge = TogetherLLMJudge(
        args.qrel,
        args.model,
        args.prompt_file,
        args.prompt_type,
        args.few_shot_count,
        use_batch=args.use_batch,
        batch_size=args.batch_size,
        batch_save_dir=args.batch_save_dir,
        keep_batch_files=keep_files,
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()
