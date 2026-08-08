"""
Curated catalog of lightweight models the user can pick from the frontend's
"Model Gallery" panel. This is intentionally a small, hand-picked list —
not a live HuggingFace Hub search — so every option is verified to be
small enough to run on modest hardware.

Adding a model is just adding an entry here; no other code changes needed
as long as it's a standard HuggingFace `transformers`-compatible
causal-LM or sentence-embedding model.
"""

LLM_CATALOG = [
    {
        "id": "microsoft/phi-2",
        "name": "Phi-2",
        "publisher": "Microsoft",
        "params": "2.7B",
        "context_window": 2048,
        "gated": False,
        "notes": "The original default. Solid general-purpose reasoning for its size.",
    },
    {
        "id": "microsoft/Phi-4-mini-instruct",
        "name": "Phi-4-mini",
        "publisher": "Microsoft",
        "params": "3.8B",
        "context_window": 4096,
        "gated": False,
        "notes": "Newer architecture, strong instruction-following. Slightly heavier than Phi-2.",
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "publisher": "Meta",
        "params": "1B",
        "context_window": 4096,
        "gated": True,
        "notes": "Fastest option here. Gated: requires an HF_TOKEN with the Meta license accepted.",
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B Instruct",
        "publisher": "Meta",
        "params": "3B",
        "context_window": 4096,
        "gated": True,
        "notes": "Good quality/speed balance. Gated: requires an HF_TOKEN with the Meta license accepted.",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2 1.7B Instruct",
        "publisher": "HuggingFace",
        "params": "1.7B",
        "context_window": 4096,
        "gated": False,
        "notes": "Compact, fast, MIT licensed. Good default for CPU-only deployments.",
    },
]

EMBEDDING_CATALOG = [
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "MiniLM-L6-v2",
        "publisher": "Sentence-Transformers",
        "params": "22M",
        "dims": 384,
        "notes": "The original default. Fast, well-rounded baseline.",
    },
    {
        "id": "BAAI/bge-small-en-v1.5",
        "name": "BGE-small-en v1.5",
        "publisher": "BAAI",
        "params": "33M",
        "dims": 384,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "notes": "Strong retrieval quality for its size. Instruction is auto-applied to queries.",
    },
    {
        "id": "intfloat/e5-small-v2",
        "name": "E5-small-v2",
        "publisher": "Intfloat",
        "params": "33M",
        "dims": 384,
        "query_instruction": "query: ",
        "text_instruction": "passage: ",
        "notes": "Needs query/passage prefixes for best results — applied automatically here.",
    },
    {
        "id": "thenlper/gte-small",
        "name": "GTE-small",
        "publisher": "Alibaba (thenlper)",
        "params": "33M",
        "dims": 384,
        "notes": "Good general-purpose compact embedding model, no special prefixes needed.",
    },
]


def get_llm_entry(model_id: str) -> dict | None:
    return next((m for m in LLM_CATALOG if m["id"] == model_id), None)


def get_embedding_entry(model_id: str) -> dict | None:
    return next((m for m in EMBEDDING_CATALOG if m["id"] == model_id), None)
