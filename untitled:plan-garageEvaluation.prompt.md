## Recommended Design

Make GaRAGe a **benchmark corpus plus evaluation set**, rather than only importing its questions.

### 1. Ingest GaRAGe grounding passages

For each GaRAGe record:

- Read the `grounding` list.
- Extract each passage as a separate logical document or chunk.
- Preserve metadata such as:
  - `sample_id`
  - passage citation marker
  - provider: `web` or `ent`
  - passage date
  - passage age
  - passage position

Do **not** index:

- `answer_generate`
- `answer_related_info`
- evaluation annotations
- the question itself

Otherwise, the benchmark would leak the answer into the retrieval corpus.

A useful internal document identity would be:

```text
garage:{sample_id}:passage:{position}
```

### 2. Keep questions as evaluation records

The current importer already maps:

- `sample_id` -> evaluation `id`
- `question` -> evaluation question
- `answer_generate` -> `reference_answer`

It should additionally preserve the expected passage IDs derived from GaRAGe’s evidence annotations.

For example:

```json
{
  "id": "sample-123",
  "question": "...",
  "reference_answer": "...",
  "expected_passages": [
    "garage:sample-123:passage:2",
    "garage:sample-123:passage:5"
  ]
}
```

### 3. Use an isolated benchmark corpus

GaRAGe evaluation should use a separate data directory or Chroma collection:

```text
data/garage_eval/
├── user_docs/
├── chroma_db/
└── eval_dataset.json
```

This prevents the existing DocMind documents from affecting results. Otherwise, the evaluation measures retrieval from a mixture of GaRAGe and unrelated documents.

A separate collection could be named:

```text
docmind_garage_eval__nomic_embed_text
```

## Evaluation Pipeline

The complete flow would be:

1. Load `GaRAGe_benchmark.jsonl`.
2. Filter records:
   - valid questions
   - answer-seeking questions
   - non-sensitive questions
3. Extract only `grounding` passages.
4. Store each passage with stable metadata and IDs.
5. Build the Chroma index.
6. Ask each GaRAGe question through the normal DocMind RAG pipeline.
7. Compare retrieved passage IDs with GaRAGe’s expected evidence.
8. Compare the generated answer with `answer_generate`.
9. Aggregate retrieval and answer-quality metrics.

## Retrieval Metrics

The current filename-based hit/MRR calculation is insufficient for GaRAGe. Add passage-level metrics:

- `retrieval_hit@k`
- `retrieval_recall@k`
- `retrieval_precision@k`
- `retrieval_mrr`
- `retrieval_ndcg@k`

Ground-truth relevance could be defined using GaRAGe annotations:

- Relevant evidence:
  - `evidence_relevant == YES`
- Answer-bearing evidence:
  - `evidence_correct == ANSWER-THE-QUESTION`

The most useful primary metric would likely be:

```text
Answer-bearing passage recall@k
```

This measures whether DocMind retrieves passages that actually support answering the question.

## Answer Metrics

Keep the existing metrics:

- Semantic similarity against `answer_generate`
- Optional local LLM judge:
  - faithfulness
  - relevance

Add or consider:

- Answer completeness
- Unsupported-claim rate
- Citation correctness
- Citation coverage
- Answerable-question accuracy

GaRAGe’s `evidence_cited` field is especially valuable for checking whether the generated answer cites the passages that should support it.

## Two Evaluation Modes

Support both modes.

### Mode A: Benchmark-faithful evaluation

Index GaRAGe’s grounding passages and evaluate retrieval against the annotated evidence.

This answers:

> Can DocMind retrieve the correct evidence and generate a good answer from the GaRAGe corpus?

### Mode B: Application evaluation

Use GaRAGe questions against the user’s own indexed documents, as the current implementation does.

This answers:

> How does DocMind respond to questions that resemble real-world GaRAGe queries using my own knowledge base?

Mode A is appropriate for measuring RAG quality. Mode B is useful for stress-testing the application but is not a strict GaRAGe benchmark because the original grounding corpus is absent.

## Important Implementation Changes

The current JSON ingestion idea should have two layers:

1. **Generic JSON ingestion**
   - Parse arbitrary JSON.
   - Convert selected fields into readable text.
   - Useful for normal application documents.

2. **GaRAGe-specific ingestion**
   - Extract grounding passages structurally.
   - Preserve passage-level IDs and annotations.
   - Generate evaluation metadata.
   - Avoid indexing benchmark answers and questions.

Using a generic `json.dumps(record)` loader for GaRAGe would work technically, but it would mix questions, answers, labels, and evidence together and create benchmark leakage.

The main additional code areas would be:

- `app/core/ingestion.py`
  - JSON support and GaRAGe passage extraction
- `app/core/indexing.py`
  - stable passage metadata and isolated collections
- `app/core/evaluation.py`
  - expected passage matching and retrieval metrics
- `app/models.py`
  - passage-level evaluation fields
- `scripts/import_garage.py`
  - corpus preparation and evaluation-set generation
- `README.md`
  - benchmark workflow and leakage warning

## Recommended First Milestone

Implement a small, reproducible benchmark mode:

- Select 50 GaRAGe records.
- Extract and index only their grounding passages.
- Use `answer_generate` as the reference answer.
- Measure:
  - passage hit@4
  - answer-bearing recall@4
  - MRR
  - semantic similarity
  - optional LLM relevance and faithfulness
- Save all raw retrieved passages and generated answers in the report.

That gives a meaningful benchmark quickly and avoids conflating GaRAGe evaluation with the current user-document chat corpus.
