import argparse
import os
from typing_extensions import Optional

from dotenv import load_dotenv
from openai import OpenAI
from retry import retry
from tqdm import tqdm

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class OpenRouterLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str = "mistralai/devstral-2512:free",
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 0,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.create_openrouter_client()

    def create_openrouter_client(self):
        """Initialize OpenRouter API client."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # OpenRouter uses OpenAI-compatible API with custom base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = self.model_name

    @retry(tries=3, delay=0.1)
    def run_openrouter_llm(self, prompt, max_new_tokens):
        """Run a single request using OpenRouter API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = self.client.chat.completions.create(
            extra_body={
                "provider": {
                    "sort": "price",  # Sort by lowest price
                    "quantizations": ["fp8"]  # Only FP8 providers
                }
            },
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
        # except Exception as e:
            # print(f"Encountered {e} for {prompt}")
            # output = ""
        return output

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        prepocess: bool,
    ):
        """Predict using LLM."""
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )

        outputs = [
            self.run_openrouter_llm(prompt, max_new_tokens) 
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
        default="mistralai/devstral-2512:free",
        help="OpenRouter model name (default: mistralai/devstral-2512:free)"
    )
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    judge = OpenRouterLLMJudge(
        args.qrel,
        args.model,
        args.prompt_file,
        args.prompt_type,
        args.few_shot_count,
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()
