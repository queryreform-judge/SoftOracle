# Outcome-Level Selection for Inference-Time LLM Query Reformulation

SoftOracle is a training-free framework for inference-time query reformulation. Instead of assuming that one reformulation strategy is always best, SoftOracle treats reformulation as a per-query decision problem: each candidate reformulation is issued to the retriever, the resulting ranked list is judged, and the ranking with the highest estimated utility is selected.

The latest paper frames the central challenge as outcome-level selection. The important question is not only how to generate better reformulations, but how to decide which retrieved outcome should be trusted for the current query.

<!-- TODO: Replace ADD_FIGURE_PATH_HERE with the final figure path. -->
![SoftOracle outcome-level selection overview](figures/pipeline.png)

## Highlights

- Models inference-time query reformulation as a finite-action decision problem over candidate ranked lists.
- Estimates ranking utility directly from retrieved evidence using LLM-judged graded relevance.
- Selects among the original query and ten LLM reformulation methods without training a selector.
- Uses de-duplicated judging so overlapping documents across reformulations are assessed once.
- Improves over fixed reformulators, QPP-based selectors, and learned routing selectors on TREC DL and BEIR.
- Closes up to 79% of the per-query oracle gap on TREC benchmarks under sparse retrieval.

## Method

For an input query, SoftOracle builds a candidate pool from the original query plus LLM-generated reformulations. Each candidate query produces a ranked list under a fixed retriever. SoftOracle then estimates the utility of each ranked list by asking an LLM judge to grade the retrieved documents against the original query.

At a high level:

1. Generate candidate queries with multiple reformulators.
2. Retrieve a ranked list for each candidate using the same backend retriever.
3. Judge the top-K retrieved documents with an LLM relevance assessor.
4. Estimate DCG-style ranking utility from the judged relevance labels.
5. Select the candidate ranked list with the highest estimated utility.

The paper evaluates pointwise and batched judging, binary and graded labels, and multiple judging depths. The strongest setting uses pointwise graded labels with top-10 judging.

## Main Results

Performance is reported as nDCG@10. The main SoftOracle result uses Qwen2.5-7B reformulations with a DeepSeek-V3 judge at top-10 judging depth. The final column is the macro-average over the six BEIR datasets.

| Method | DL19 | DL20 | DL-Hard | SciFact | Arguana | COVID | FiQA | DBPedia | News | BEIR Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.506 | 0.480 | 0.285 | 0.679 | 0.397 | 0.595 | 0.236 | 0.318 | 0.395 | 0.437 |
| BM25 + RM3 | 0.515 | 0.492 | 0.264 | 0.646 | 0.380 | 0.593 | 0.192 | 0.308 | 0.426 | 0.424 |
| DPR | 0.622 | 0.653 | - | 0.318 | 0.175 | 0.332 | 0.295 | 0.263 | 0.161 | 0.257 |
| ANCE | 0.645 | 0.646 | 0.334 | 0.570 | 0.415 | 0.654 | 0.300 | 0.281 | 0.382 | 0.434 |
| ContrieverFT | 0.621 | 0.632 | - | 0.677 | 0.446 | 0.596 | 0.329 | 0.413 | 0.428 | 0.482 |
| GenQR | 0.426 | 0.420 | 0.212 | 0.692 | 0.434 | 0.652 | 0.204 | 0.288 | 0.430 | 0.450 |
| GenQR Ensemble | 0.446 | 0.488 | 0.246 | 0.704 | 0.419 | 0.678 | 0.208 | 0.346 | 0.437 | 0.465 |
| Query2E | 0.573 | 0.543 | 0.299 | 0.697 | 0.405 | 0.695 | 0.245 | 0.348 | 0.450 | 0.473 |
| QA-expand | 0.569 | 0.546 | 0.293 | 0.686 | 0.394 | 0.673 | 0.223 | 0.334 | 0.434 | 0.457 |
| CSQE | 0.680 | 0.610 | 0.338 | 0.718 | 0.401 | 0.676 | 0.220 | 0.377 | 0.450 | 0.474 |
| Query2Doc (CoT) | 0.603 | 0.575 | 0.307 | 0.710 | 0.401 | 0.700 | 0.241 | 0.367 | 0.435 | 0.476 |
| Query2Doc (FS) | 0.599 | 0.567 | 0.297 | 0.715 | 0.398 | 0.742 | 0.243 | 0.386 | 0.478 | 0.494 |
| Query2Doc (ZS) | 0.618 | 0.585 | 0.323 | 0.704 | 0.401 | 0.707 | 0.246 | 0.384 | 0.451 | 0.482 |
| LameR | 0.657 | 0.632 | 0.347 | 0.714 | 0.406 | 0.696 | 0.234 | 0.390 | 0.442 | 0.480 |
| MuGi | 0.641 | 0.616 | 0.313 | 0.706 | 0.393 | 0.677 | 0.237 | 0.399 | 0.444 | 0.476 |
| Pre-QPP (Best) | 0.653 | 0.638 | 0.320 | 0.718 | 0.432 | 0.678 | 0.239 | 0.400 | 0.458 | 0.488 |
| Post-QPP (Best) | 0.644 | 0.597 | 0.343 | 0.709 | 0.427 | 0.650 | 0.236 | 0.384 | 0.453 | 0.477 |
| Neural-QPP (Best) | 0.656 | 0.646 | 0.333 | 0.716 | 0.417 | 0.761 | 0.258 | 0.389 | 0.457 | 0.499 |
| SW-Ranking | 0.622 | 0.590 | 0.332 | 0.709 | 0.400 | 0.706 | 0.234 | 0.372 | 0.429 | 0.475 |
| MF | 0.593 | 0.575 | 0.327 | 0.696 | 0.396 | 0.661 | 0.235 | 0.363 | 0.429 | 0.463 |
| BERT classifier | 0.685 | 0.609 | 0.332 | 0.718 | 0.401 | 0.676 | 0.220 | 0.375 | 0.451 | 0.473 |
| Random | 0.522 | 0.497 | 0.293 | 0.698 | 0.406 | 0.709 | 0.230 | 0.353 | 0.459 | 0.476 |
| **SoftOracle (Ours)** | **0.697** | **0.695** | **0.370** | **0.723** | **0.434** | 0.756 | **0.275** | **0.422** | **0.487** | **0.517** |

Key takeaways from the latest result:

- SoftOracle improves over the strongest fixed reformulator by 2.5%, 10.0%, and 6.6% on DL19, DL20, and DL-Hard.
- SoftOracle improves over the strongest QPP selectors by 6.3% to 7.9% across TREC DL benchmarks.
- On BEIR, SoftOracle reaches the strongest macro-average performance at 0.517.
- The largest gains appear when reformulations induce diverse ranked lists, especially on DL20 and DL-Hard.

## LLM Judge and Reformulator Effects

Judge quality matters more than reformulator size for outcome-level selection. With Qwen2.5-7B reformulations, replacing DeepSeek-V3 as judge with Qwen2.5-7B reduces DL20 from 0.695 to 0.655 and DL-Hard from 0.370 to 0.336.

| Reformulator | Judge | DL19 | DL20 | DL-Hard |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B | Best single | 0.680 | 0.632 | 0.347 |
| Qwen2.5-7B | DeepSeek-V3 | 0.697 | 0.695 | 0.370 |
| Qwen2.5-7B | Qwen2.5-7B | 0.684 | 0.655 | 0.336 |
| Qwen2.5-7B | Llama-3.3-70B | 0.671 | 0.634 | 0.336 |
| Llama-3.3-70B | Best single | 0.671 | 0.664 | 0.367 |
| Llama-3.3-70B | DeepSeek-V3 | 0.705 | 0.687 | 0.374 |
| Llama-3.3-70B | Qwen2.5-7B | 0.696 | 0.655 | 0.360 |
| Llama-3.3-70B | Llama-3.3-70B | 0.666 | 0.642 | 0.345 |
| DeepSeek-V3 | Best single | 0.709 | 0.648 | 0.364 |
| DeepSeek-V3 | DeepSeek-V3 | 0.709 | 0.699 | 0.362 |
| DeepSeek-V3 | Qwen2.5-7B | 0.699 | 0.665 | 0.354 |
| DeepSeek-V3 | Llama-3.3-70B | 0.693 | 0.632 | 0.353 |

## Judging Strategy

The latest paper finds that pointwise graded relevance judgments are the most reliable utility signal.

| Judge strategy | DL19 | DL20 | DL-Hard |
| --- | ---: | ---: | ---: |
| Pointwise, graded (0-3) | **0.697** | **0.695** | **0.370** |
| Pointwise, binary (0-1) | 0.676 | 0.599 | 0.323 |
| Batched, graded (0-3) | 0.676 | 0.651 | 0.351 |
| Batched, binary (0-1) | 0.660 | 0.617 | 0.320 |

Increasing judging depth also helps. Moving from K=3 to K=10 improves DL19 from 0.674 to 0.697, DL20 from 0.655 to 0.695, and DL-Hard from 0.356 to 0.370.

## Efficiency

With ten reformulation candidates plus the original query and K=10, naive pointwise judging requires 110 document assessments per query. Because the same retrieved document often appears under multiple reformulations, SoftOracle reuses document judgments across candidate lists. In the paper, de-duplicated judging reduces the average to 33.7 unique assessments per query. Smaller judging depths reduce the average further to 18.4 for K=5 and 11.8 for K=3.

## Experimental Setup

Datasets:

| Dataset | Queries | Documents |
| --- | ---: | ---: |
| TREC DL19 | 43 | 8,841,823 |
| TREC DL20 | 54 | 8,841,823 |
| DL-Hard | 50 | 8,841,823 |
| SciFact | 300 | 5,183 |
| Arguana | 1,406 | 8,674 |
| TREC-COVID | 50 | 171,332 |
| FiQA | 648 | 57,638 |
| DBPedia | 400 | 4,635,922 |
| TREC-News | 57 | 594,977 |

Reformulators:

- GenQR
- GenQR-Ensemble
- QA-expand
- Query2E
- Query2Doc zero-shot, few-shot, and chain-of-thought
- CSQE
- LameR
- MuGI

Selector baselines:

- Fixed selectors: original query and best single reformulator.
- Pre-retrieval QPP selectors.
- Post-retrieval QPP selectors.
- Neural QPP selectors.
- Learned routing selectors: SW-Ranking, matrix factorization, and BERT classifier.

## Repository Layout

```text
SoftOracle/
|-- src/
|   |-- querygym/          # QueryGym integration and reformulation methods
|   `-- umbrela/           # LLM judge utilities and scoring scripts
|-- baseline_querygym/     # Reformulation outputs and baseline retrieval results
|-- llm-judge-feedback/    # LLM judge outputs
|-- figures/               # Figures used by the paper and README
`-- README.md
```

## Citation

Citation metadata will be updated when the paper version is finalized.

```bibtex
@inproceedings{softoracle2026,
  title = {Outcome-Level Selection for Inference-Time LLM Query Reformulation},
  author = {Anonymous},
  booktitle = {},
  year = {2026}
}
```
