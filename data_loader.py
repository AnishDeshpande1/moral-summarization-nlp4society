"""
Data loader for the political-fairness audit of the moral-summarization paper.

Loads the precomputed metric pickles in ``results/automated_evaluation/``,
filters out the eval-pipeline summary rows (``mean`` / ``std``), normalizes the
``QaFactEval`` vs ``QAFactEval`` key inconsistency, parses political leaning out
of AllSides article IDs (``allsides_<topic>_<l|c|r>_<n>``), and returns tidy
in-memory DataFrames.

This module is import-only -- it does not write any files. Use the functions
from a notebook:

    from data_loader import build_full_long, build_wide_by_method
    long_df = build_full_long()
    wide_df = build_wide_by_method(long_df)

Each pickle is a ``dict[metric_name -> DataFrame]``. The DataFrames are
MultiIndexed by ``(model, dataset, article)`` and have one column per prompting
method (``vanilla``, ``simple``, ``cot``, ``oracle``, ``class``) plus
``original`` (the reference value computed on the source article).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
PICKLE_DIR = REPO_ROOT / "results" / "automated_evaluation"

PICKLE_FILES = [
    "llama_345_test_set.pickle",
    "llama_commandr.pickle",
    "deepseek.pickle",
    "llama3.1-8b-instruct-q4_K_M.pickle",
]

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
# Original 5 methods — used by Llama-70B, Command-R, DeepSeek
PROMPTING_METHODS = ["vanilla", "simple", "cot", "oracle", "class"]

# 4 new few-shot methods — used by Llama-3.1-8B
FEWSHOT_METHODS = ["simple_fewshot", "cot_fewshot",
                   "simple_fewshot_mft", "cot_fewshot_mft"]

# All 9 methods combined — use this when analysing the Llama-8B model
ALL_PROMPTING_METHODS = PROMPTING_METHODS + FEWSHOT_METHODS

PAPER_NAME_MAP = {
    "vanilla": "Plain",
    "simple": "Direct",
    "cot": "CoT",
    "oracle": "Oracle",
    "class": "Class",
    "simple_fewshot": "Few-Shot Direct",
    "cot_fewshot": "Few-Shot CoT",
    "simple_fewshot_mft": "Few-Shot Direct+MFT",
    "cot_fewshot_mft": "Few-Shot CoT+MFT",
}

# Metric-name normalization (one pickle uses QaFactEval, others QAFactEval).
METRIC_RENAME = {"QaFactEval": "QAFactEval"}

# AllSides article-ID pattern: allsides_<topic>_<l|c|r>_<n>.
# Topic itself may contain underscores ("gun_control_and_gun_rights"), so we
# anchor on the leaning letter + final numeric index.
ALLSIDES_RE = re.compile(r"^allsides_(?P<topic>.+)_(?P<leaning>[lcr])_(?P<idx>\d+)$")
LEANING_MAP = {"l": "left", "c": "center", "r": "right"}

# Summary labels appended by the eval pipeline that we always drop.
SUMMARY_ROW_LABELS = {"mean", "std", "mean_test", "std_test", "all"}


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_pickle(path: Path) -> dict[str, pd.DataFrame]:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_metric_keys(d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Rename ``QaFactEval`` -> ``QAFactEval`` so all sources agree."""
    return {METRIC_RENAME.get(k, k): v for k, v in d.items()}


def drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``mean``/``std`` summary rows from the MultiIndex."""
    mask = ~df.index.to_frame().isin(SUMMARY_ROW_LABELS).any(axis=1)
    return df.loc[mask]


# ------------------------------------------------------------------
# Reshaping
# ------------------------------------------------------------------
def melt_metric_df(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Convert one metric DataFrame to tidy long format."""
    df = df.copy().reset_index()
    long = df.melt(
        id_vars=["model", "dataset", "article"],
        var_name="prompting_method",
        value_name="value",
    )
    long["metric"] = metric_name
    return long


def parse_allsides_id(article_id: str) -> dict[str, str | None]:
    """Extract topic, leaning, replicate from an AllSides article ID."""
    m = ALLSIDES_RE.match(article_id)
    if not m:
        return {"topic": None, "leaning": None, "replicate": None}
    return {
        "topic": m.group("topic"),
        "leaning": LEANING_MAP[m.group("leaning")],
        "replicate": m.group("idx"),
    }


def annotate_allsides(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add topic/leaning/replicate columns (only meaningful for AllSides rows)."""
    parsed = long_df["article"].apply(parse_allsides_id).apply(pd.Series)
    return pd.concat([long_df, parsed], axis=1)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
def build_long_for_pickle(pickle_path: Path) -> pd.DataFrame:
    """Load one pickle and return all of its metrics in tidy long format."""
    raw = normalize_metric_keys(load_pickle(pickle_path))
    pieces: list[pd.DataFrame] = []
    for metric_name, df in raw.items():
        df = drop_summary_rows(df)
        pieces.append(melt_metric_df(df, metric_name))
    long = pd.concat(pieces, ignore_index=True)
    long["source_pickle"] = pickle_path.name
    return long


def deduplicate_across_pickles(long: pd.DataFrame) -> pd.DataFrame:
    """A given (model, dataset, article, prompting_method, metric) cell can
    appear in more than one pickle. Keep the row from the pickle with the most
    articles for that (model, dataset) combination -- i.e. prefer the
    full-coverage source over the small test-set source.
    """
    coverage = (
        long.groupby(["model", "dataset", "source_pickle"])["article"]
        .nunique()
        .reset_index(name="n_articles")
    )
    top = (
        coverage.sort_values("n_articles", ascending=False)
        .drop_duplicates(["model", "dataset"])
        [["model", "dataset", "source_pickle"]]
        .rename(columns={"source_pickle": "preferred_source"})
    )
    merged = long.merge(top, on=["model", "dataset"], how="left")
    merged = merged[merged["source_pickle"] == merged["preferred_source"]].copy()
    return merged.drop(columns=["preferred_source"])


def build_full_long() -> pd.DataFrame:
    """Return the master tidy long-format DataFrame across all three pickles.

    Columns: model, dataset, article, topic, leaning, replicate,
    prompting_method, prompting_method_paper, metric, value, source_pickle.
    """
    pieces = [build_long_for_pickle(PICKLE_DIR / f) for f in PICKLE_FILES]
    long = pd.concat(pieces, ignore_index=True)
    long = deduplicate_across_pickles(long)
    long = annotate_allsides(long)
    long["prompting_method_paper"] = long["prompting_method"].map(PAPER_NAME_MAP)
    cols = [
        "model",
        "dataset",
        "article",
        "topic",
        "leaning",
        "replicate",
        "prompting_method",
        "prompting_method_paper",
        "metric",
        "value",
        "source_pickle",
    ]
    return long[cols]


def build_wide_by_method(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long DataFrame so each prompting method is its own column.

    One row per (model, dataset, article, metric). The ``original`` column
    holds the reference value (per-article counterpart, useful as a
    denominator for retention rates).

    We pivot on a minimal index and merge topic/leaning back in afterwards;
    including them in the pivot index produces a cartesian product across the
    non-AllSides datasets where those fields are NaN.
    """
    wide = (
        long.pivot_table(
            index=["model", "dataset", "article", "metric"],
            columns="prompting_method",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    article_meta = (
        long[["article", "topic", "leaning"]]
        .drop_duplicates(subset=["article"])
    )
    wide = wide.merge(article_meta, on="article", how="left")

    front = ["model", "dataset", "article", "topic", "leaning", "metric"]
    method_cols = [c for c in ALL_PROMPTING_METHODS + ["original"] if c in wide.columns]
    return wide[front + method_cols]


# ------------------------------------------------------------------
# Convenience helpers (handy from the notebook)
# ------------------------------------------------------------------
def allsides_only(long: pd.DataFrame) -> pd.DataFrame:
    """Return just the AllSides rows -- the slice with political-leaning labels."""
    return long[long["dataset"] == "allsides"].copy()


def summary_counts(long: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: number of unique articles per (model, leaning) for AllSides."""
    return (
        allsides_only(long)
        .groupby(["model", "leaning"])["article"]
        .nunique()
        .unstack("leaning", fill_value=0)
        .reset_index()
    )
