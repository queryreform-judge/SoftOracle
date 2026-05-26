import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing_extensions import Optional

from dotenv import load_dotenv
from openai import OpenAI
from retry import retry
from tqdm import tqdm

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1]


class OSLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 0,
        vllm_port: int = 1234,
        vllm_host: str = "localhost",
        batch_size: int = 1,
        max_workers: int = 10,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.vllm_port = vllm_port
        self.vllm_host = vllm_host
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.create_vllm_client()

    def create_vllm_client(self):
        # vLLM serves an OpenAI-compatible API
        base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        self.client = OpenAI(
            api_key="dummy-key",  # vLLM doesn't require a real API key
            base_url=base_url,
        )
        self.engine = self.model_name

    @retry(tries=3, delay=0.1)
    def run_os_llm(self, prompt, max_new_tokens):
        if 'gemma' in self.model_name:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

        try:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages,
                max_tokens=2048,
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

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
    ):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )

        # Use batch processing if batch_size > 1
        if self.batch_size > 1 and len(self.prompts) > 1:
            outputs = self.run_batch_requests(self.prompts, max_new_tokens)
        else:
            outputs = [
                self.run_os_llm(prompt, max_new_tokens) for prompt in tqdm(self.prompts)
            ]
        return outputs

    def run_batch_requests(self, prompts, max_new_tokens):
        """Run multiple requests concurrently using ThreadPoolExecutor."""
        print(f"Processing {len(prompts)} requests with batch_size={self.batch_size}, max_workers={self.max_workers}")
        
        outputs = [None] * len(prompts)  # Pre-allocate to maintain order
        
        # Process in batches to control memory usage
        for batch_start in range(0, len(prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all requests in this batch
                future_to_index = {
                    executor.submit(self.run_os_llm, prompt, max_new_tokens): idx
                    for idx, prompt in zip(batch_indices, batch_prompts)
                }
                
                # Collect results as they complete, maintaining order
                for future in tqdm(
                    as_completed(future_to_index),
                    total=len(batch_prompts),
                    desc=f"Batch {batch_start//self.batch_size + 1}",
                    leave=False
                ):
                    idx = future_to_index[future]
                    try:
                        outputs[idx] = future.result()
                    except Exception as e:
                        print(f"Error processing prompt at index {idx}: {e}")
                        outputs[idx] = ""
        
        return outputs

    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
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
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--vllm_port", type=int, default=1234, help="Port for vLLM server")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="Host for vLLM server")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of requests to process in each batch (default: 1, sequential)")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of concurrent workers for batch processing (default: 10)")
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    judge = OSLLMJudge(
        args.qrel, 
        args.model, 
        args.prompt_file, 
        args.prompt_type, 
        args.few_shot_count,
        vllm_port=args.vllm_port,
        vllm_host=args.vllm_host,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()

