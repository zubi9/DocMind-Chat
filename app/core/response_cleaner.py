"""
Cleans raw LLM completions before they're returned to the frontend.

Two backends means two different failure modes for "messy" text:

  - Ollama (no chat endpoint used here — llama-index calls /api/generate):
    Ollama applies the model's chat template server-side, but with no
    output-length cap (now fixed in llm_setup.py) or explicit stop
    sequences, small instruct models can keep generating past the answer
    into a fabricated new turn ("\\n\\nQuestion: ...\\nAnswer: ...").
  - HuggingFace: occasionally leaks literal special tokens
    (<s>, [INST], <<SYS>>, <|im_start|>, etc.) into the decoded text,
    depending on tokenizer/model combination and prompt template used.

This module is a defensive, conservative cleanup pass — it only strips
recognizable artifacts and trims runaway continuations, it never rewrites
or summarizes the actual answer content.
"""
import logging
import re

logger = logging.getLogger("docmind.response_cleaner")

# Special/control tokens that occasionally leak into decoded text across
# Llama/Mistral/Phi/ChatML-style instruct models.
_SPECIAL_TOKEN_PATTERNS = [
    r"</?s>",
    r"\[/?INST\]",
    r"<<SYS>>|<</SYS>>",
    r"<\|im_start\|>|<\|im_end\|>",
    r"<\|system\|>|<\|user\|>|<\|assistant\|>|<\|end\|>",
    r"<\|endoftext\|>",
]
_SPECIAL_TOKEN_RE = re.compile("|".join(_SPECIAL_TOKEN_PATTERNS))

# If the model continues past the real answer into a fabricated new turn,
# it almost always does so with one of these markers at the start of a
# line. Cut everything from the first such marker onward. Conservative on
# purpose: only matches at the start of a line, and only after at least
# some real answer content has already been produced.
_RUNAWAY_TURN_RE = re.compile(
    r"\n\s*(Question|Query|User|Human)\s*:\s*",
    re.IGNORECASE,
)

# Leading echo of the QA prompt template's own "Answer:" label, if the
# model repeats it verbatim instead of just answering.
_LEADING_ANSWER_LABEL_RE = re.compile(r"^\s*Answer\s*:\s*", re.IGNORECASE)


def clean_response(text: str) -> str:
    """Strips template artifacts and trims runaway continuations from raw LLM output."""
    if not text:
        return text

    original_len = len(text)
    cleaned = _SPECIAL_TOKEN_RE.sub("", text)
    cleaned = _LEADING_ANSWER_LABEL_RE.sub("", cleaned)

    match = _RUNAWAY_TURN_RE.search(cleaned)
    if match and match.start() > 20:  # keep at least a bit of real content before trimming
        cleaned = cleaned[: match.start()]

    # Collapse 3+ consecutive newlines (common after stripping tokens that
    # sat alone on their own line) down to a single paragraph break.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) < original_len * 0.5 and original_len > 200:
        # We trimmed away more than half the response — log it so this is
        # visible in practice rather than silently discarding content if
        # the heuristic above is ever too aggressive for a given model.
        logger.info(
            "clean_response trimmed response from %d to %d chars (runaway-turn or token cleanup).",
            original_len, len(cleaned),
        )

    return cleaned
