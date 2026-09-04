# DocMind Chat RAG Evaluation Report

**Evaluation date:** 2026-09-04  
**Dataset:** GaRAGe benchmark, reproducible 50-question subset  
**Source selection:** `--limit 50 --seed 42`  
**Grounding corpus:** 750 GaRAGe passages  
**Backend:** Ollama  
**Generation model:** `phi3:mini`  
**Embedding model:** `nomic-embed-text`

## Objective

Measure the retrieval and answer-generation performance of DocMind Chat using GaRAGe questions and grounding passages. Only grounding passages were indexed; GaRAGe questions, reference answers, and evaluation labels were excluded from the searchable corpus to prevent answer leakage.

## Results

| Metric | Result |
|---|---:|
| Questions evaluated | 50 |
| Query errors | 0 |
| Average latency | 1.542 s |
| Retrieval hit rate | 83.72% |
| Retrieval MRR | 0.7694 |
| Answer-bearing passage recall | 35.02% |
| Answer-bearing passage precision | 61.05% |
| Retrieval nDCG | 0.7460 |
| Answer semantic similarity | 0.8110 |

Retrieval metrics were available for 43 questions with annotated answer-bearing passages. Semantic similarity was available for all 50 questions. The index remained synchronized for every evaluated query.

## Metric Definitions

- **Retrieval hit rate:** Percentage of questions where at least one answer-bearing passage was retrieved.
- **MRR:** Reciprocal rank of the first retrieved answer-bearing passage, averaged across eligible questions.
- **Passage recall:** Fraction of annotated answer-bearing passages retrieved.
- **Passage precision:** Fraction of unique retrieved passages that were answer-bearing.
- **nDCG:** Rank-sensitive retrieval quality, rewarding answer-bearing passages appearing earlier in the result list.
- **Semantic similarity:** Cosine similarity between the generated answer and GaRAGe's `answer_generate`, using the active embedding model.
- **Average latency:** End-to-end time for retrieval and answer generation per question.

## Interpretation

DocMind retrieved at least one answer-bearing passage for most eligible questions, with an 83.72% hit rate and 0.7694 MRR. The lower 35.02% passage recall indicates that the system often found some useful evidence but did not retrieve all answer-bearing passages required by the annotations. Precision of 61.05% shows that a meaningful portion of the retrieved context was relevant, while some retrieved passages were unrelated or redundant.

The semantic similarity score of 0.8110 indicates generally strong alignment between generated answers and the reference answers. This should be interpreted together with retrieval metrics: high answer similarity does not prove that every claim is supported by the retrieved evidence.

## Scope and Limitations

- This report covers the configured 50-question subset, not all 780 eligible GaRAGe records.
- LLM-judge faithfulness and relevance were not enabled, so no independent hallucination or answer-relevance scores are reported.
- Keyword coverage was not configured for GaRAGe items.
- Retrieval ground truth uses passages labeled `evidence_correct == ANSWER-THE-QUESTION`.
- Semantic similarity is an embedding-based proxy, not human evaluation.
- The run used the local `phi3:mini` model; results may change with a different generation model, embedding model, chunk size, or retrieval configuration.

## Reproduction

From the repository root:

```bash
docker compose \
  -f docker-compose.ollama.yml \
  -f docker-compose.garage.yml \
  up -d --build

curl -X POST http://localhost:8000/embeddings/build

python3 scripts/evaluate.py \
  --base-url http://localhost:8000 \
  --save
```

The raw report is available at:

`eval_results/eval-20260904-202522.json`
