"""
Build the corpus of texts (articles + summaries) for the political stance-shift audit.

Walks ``results/summaries/allsides/<article_id>/``, extracts the source article body
from each ``vanilla_prompt.txt`` (one row per article, deduplicated) and every summary
body from ``<method>_response_<model>_<seed>.txt`` files, and writes a tidy parquet at
``political_shift/data/texts.parquet``.

This script does *not* run any model inference. Token counts use ``bert-base-cased``
since the downstream classifier (``bucketresearch/politicalBiasBERT``) shares that
tokenizer.

Run from the repo root:

    python -m political_shift.build_texts
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARIES_DIR = REPO_ROOT / "results" / "summaries" / "allsides"
OUT_PATH = REPO_ROOT / "political_shift" / "data" / "texts.parquet"

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TOKENIZER_NAME = "bert-base-cased"

# AllSides article-ID pattern -- mirrors data_loader.ALLSIDES_RE.
ALLSIDES_RE = re.compile(r"^allsides_(?P<topic>.+)_(?P<leaning>[lcr])_(?P<idx>\d+)$")
LEANING_MAP = {"l": "left", "c": "center", "r": "right"}

# Strip prompt/response scaffolding.
ARTICLE_RE = re.compile(
    r"Here is the news article:\s*(.*?)\s*The summary has to be returned",
    re.DOTALL,
)
SUMMARY_RE = re.compile(r"SUMMARY:\s*(.*?)\s*END OF SUMMARY\.", re.DOTALL)


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------
def parse_article_id(article_id: str) -> tuple[str | None, str | None]:
    """Return ``(topic, leaning)`` for an AllSides article ID, or ``(None, None)``."""
    m = ALLSIDES_RE.match(article_id)
    if not m:
        return None, None
    return m.group("topic"), LEANING_MAP[m.group("leaning")]


def extract_article_body(prompt_path: Path) -> str | None:
    """Pull the article body out of a ``vanilla_prompt.txt`` scaffold."""
    text = prompt_path.read_text(encoding="utf-8", errors="replace")
    m = ARTICLE_RE.search(text)
    return m.group(1).strip() if m else None


def extract_summary_body(response_path: Path) -> str | None:
    """Pull the summary body out of a model response file.

    Returns ``None`` only if no ``SUMMARY: ... END OF SUMMARY.`` block is present.
    """
    text = response_path.read_text(encoding="utf-8", errors="replace")
    m = SUMMARY_RE.search(text)
    return m.group(1).strip() if m else None


def parse_response_filename(fname: str) -> tuple[str, str, str] | None:
    """Parse ``<method>_response_<model>_<seed>.txt`` into ``(method, model, seed)``.

    Model names contain hyphens but no ``_response_`` infix, so we split on that
    sentinel and then peel the trailing seed off the right.
    """
    stem = fname.removesuffix(".txt")
    if "_response_" not in stem:
        return None
    method, _, rest = stem.partition("_response_")
    model, _, seed = rest.rpartition("_")
    if not method or not model or not seed:
        return None
    return method, model, seed


# ------------------------------------------------------------------
# Main build
# ------------------------------------------------------------------
def build_rows() -> list[dict]:
    """Walk the summaries tree and produce one row per article / summary."""
    rows: list[dict] = []

    article_dirs = sorted(p for p in SUMMARIES_DIR.iterdir() if p.is_dir())
    for article_dir in article_dirs:
        article_id = article_dir.name
        topic, leaning = parse_article_id(article_id)

        # (a) Article body -- one row per article, deduplicated.
        vanilla_prompt = article_dir / "vanilla_prompt.txt"
        if vanilla_prompt.exists():
            article_body = extract_article_body(vanilla_prompt)
            if article_body:
                rows.append(
                    {
                        "text_id": f"art:{article_id}",
                        "kind": "article",
                        "article_id": article_id,
                        "leaning": leaning,
                        "topic": topic,
                        "method": None,
                        "model": None,
                        "seed": None,
                        "text": article_body,
                    }
                )

        # (b) Every model summary for this article.
        for response_path in sorted(article_dir.glob("*_response_*.txt")):
            parsed = parse_response_filename(response_path.name)
            if parsed is None:
                continue
            method, model, seed = parsed
            summary_body = extract_summary_body(response_path)
            if not summary_body:
                continue
            rows.append(
                {
                    "text_id": f"sum:{article_id}:{method}:{model}:{seed}",
                    "kind": "summary",
                    "article_id": article_id,
                    "leaning": leaning,
                    "topic": topic,
                    "method": method,
                    "model": model,
                    "seed": seed,
                    "text": summary_body,
                }
            )

    return rows


def add_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``n_tokens`` column using the ``bert-base-cased`` tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    encodings = tokenizer(
        df["text"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    df["n_tokens"] = [len(ids) for ids in encodings["input_ids"]]
    return df


def main() -> None:
    if not SUMMARIES_DIR.exists():
        raise FileNotFoundError(f"Summaries directory not found: {SUMMARIES_DIR}")

    rows = build_rows()
    df = pd.DataFrame(rows)

    # Deduplicate articles defensively (one vanilla_prompt per article folder, but be safe).
    df = df.drop_duplicates(subset=["text_id"]).reset_index(drop=True)

    df = add_token_counts(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    n_articles = int((df["kind"] == "article").sum())
    n_summaries = int((df["kind"] == "summary").sum())
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(f"  articles : {n_articles:,}")
    print(f"  summaries: {n_summaries:,}")
    print(
        "  n_tokens stats: "
        f"min={df['n_tokens'].min()}, "
        f"median={int(df['n_tokens'].median())}, "
        f"max={df['n_tokens'].max()}"
    )
    over_512 = int((df["n_tokens"] > 512).sum())
    print(f"  texts > 512 tokens (will be chunked): {over_512:,}")


if __name__ == "__main__":
    main()
