"""
Score every text in ``texts.parquet`` with ``bucketresearch/politicalBiasBERT``.

Long texts (> 512 BERT tokens) are split into overlapping 512-token windows (stride
128); softmax probabilities are averaged across windows. Summaries already fit in a
single window, so chunking only fires on full article bodies.

Two scalar scores are derived from the averaged probabilities:
  ``direction    = P(right) - P(left)``  in [-1, +1]; +1 = right-leaning, -1 = left-leaning
  ``polarization = 1 - P(center)``       in [0, 1];   0 = fully centrist

Results are appended to ``political_shift/data/stance_scores.parquet`` every batch
(default 32 texts). On startup any already-scored ``text_id`` is skipped, so the
script is fully resumable -- if it dies mid-run you just re-launch it.

Usage::

    python -m political_shift.score_texts                  # full corpus
    python -m political_shift.score_texts --limit 20       # smoke test
    python -m political_shift.score_texts --batch-size 16  # smaller batch
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ------------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TEXTS_PATH = REPO_ROOT / "political_shift" / "data" / "texts.parquet"
SCORES_PATH = REPO_ROOT / "political_shift" / "data" / "stance_scores.parquet"

TOKENIZER_NAME = "bert-base-cased"
MODEL_NAME = "bucketresearch/politicalBiasBERT"

MAX_LENGTH = 512
STRIDE = 128


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Chunking + scoring
# ------------------------------------------------------------------
def encode_with_chunks(
    text: str,
    tokenizer: AutoTokenizer,
    needs_chunking: bool,
) -> dict[str, torch.Tensor]:
    """Tokenize ``text`` into one or more 512-token windows.

    ``needs_chunking=False`` is a fast path for texts known to fit in a single
    window (i.e. summaries), avoiding the overflow bookkeeping cost.
    """
    if not needs_chunking:
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    # Overflow chunks have varying lengths (the tail chunk is shorter); pad them
    # to MAX_LENGTH so they stack into a single batched tensor.
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding="max_length",
    )


@torch.no_grad()
def score_text(
    text: str,
    needs_chunking: bool,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return ``(mean_probs[3], n_chunks)`` for one text."""
    enc = encode_with_chunks(text, tokenizer, needs_chunking)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = logits.softmax(dim=-1)  # (n_chunks, 3)
    return probs.mean(dim=0).cpu(), int(probs.shape[0])


def direction_from_probs(mean_probs: torch.Tensor) -> float:
    """P(right) - P(left), in [-1, +1]. Positive = right-leaning, negative = left-leaning."""
    return float((mean_probs[2] - mean_probs[0]).item())


def polarization_from_probs(mean_probs: torch.Tensor) -> float:
    """1 - P(center), in [0, 1]. 0 = fully centrist."""
    return float((1.0 - mean_probs[1]).item())


# ------------------------------------------------------------------
# Resumable I/O
# ------------------------------------------------------------------
def load_existing_scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        return pd.DataFrame(
            columns=[
                "text_id",
                "p_left",
                "p_center",
                "p_right",
                "direction",
                "polarization",
                "n_chunks",
                "scored_at",
            ]
        )
    try:
        return pd.read_parquet(SCORES_PATH)
    except Exception as e:
        print(f"WARN: failed to read existing scores ({e}); starting fresh.")
        return pd.DataFrame(
            columns=[
                "text_id",
                "p_left",
                "p_center",
                "p_right",
                "direction",
                "polarization",
                "n_chunks",
                "scored_at",
            ]
        )


def append_scores(new_rows: list[dict], existing: pd.DataFrame) -> pd.DataFrame:
    """Concatenate ``new_rows`` onto ``existing`` and rewrite the parquet."""
    new_df = pd.DataFrame(new_rows)
    merged = pd.concat([existing, new_df], ignore_index=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Parquet doesn't support in-place append; we rewrite the file. The whole
    # corpus is small (~thousands of rows), so this is cheap and keeps the
    # output format simple (single file, no manifest).
    merged.to_parquet(SCORES_PATH, index=False)
    return merged


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most this many *new* texts (for smoke-testing).",
    )
    args = parser.parse_args()

    if not TEXTS_PATH.exists():
        raise FileNotFoundError(
            f"texts.parquet not found at {TEXTS_PATH}. Run build_texts.py first."
        )

    texts_df = pd.read_parquet(TEXTS_PATH)
    print(f"Loaded {len(texts_df):,} texts from {TEXTS_PATH}")

    existing = load_existing_scores()
    done_ids: set[str] = set(existing["text_id"].tolist())
    print(f"Already scored : {len(done_ids):,}")

    todo_df = texts_df[~texts_df["text_id"].isin(done_ids)].reset_index(drop=True)
    if args.limit is not None:
        todo_df = todo_df.head(args.limit)
    print(f"Scoring        : {len(todo_df):,}")

    if len(todo_df) == 0:
        print("Nothing to do. Existing scores:")
        print(f"  {SCORES_PATH}  ({len(existing):,} rows)")
        return

    device = pick_device()
    print(f"Device         : {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    batch_size = args.batch_size
    batch_rows: list[dict] = []
    n_batches = (len(todo_df) + batch_size - 1) // batch_size
    t0 = time.time()

    for i, row in enumerate(todo_df.itertuples(index=False), start=1):
        needs_chunking = bool(row.n_tokens > MAX_LENGTH)
        mean_probs, n_chunks = score_text(
            row.text, needs_chunking, tokenizer, model, device
        )
        batch_rows.append(
            {
                "text_id": row.text_id,
                "p_left": float(mean_probs[0].item()),
                "p_center": float(mean_probs[1].item()),
                "p_right": float(mean_probs[2].item()),
                "direction": direction_from_probs(mean_probs),
                "polarization": polarization_from_probs(mean_probs),
                "n_chunks": n_chunks,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if len(batch_rows) >= batch_size or i == len(todo_df):
            existing = append_scores(batch_rows, existing)
            elapsed = time.time() - t0
            done_so_far = i
            rate = done_so_far / elapsed if elapsed > 0 else 0.0
            batch_idx = (i + batch_size - 1) // batch_size
            print(
                f"  batch {batch_idx:>3}/{n_batches} "
                f"({done_so_far:>5}/{len(todo_df)}) "
                f"{rate:5.1f} texts/s -> {SCORES_PATH.name} "
                f"now {len(existing):,} rows"
            )
            batch_rows = []

    print(f"\nDone. {len(existing):,} total rows in {SCORES_PATH}")


if __name__ == "__main__":
    main()
