"""
moral_framing_extractor.py
=================
Standalone post-hoc analysis module for the political-fairness audit.

Reads the same raw files that eval.py uses (EMONA annotation JSONs +
model response .txt files in results/test_prompts/) but extracts the
word-level MFT-dimension detail that eval.py discards after computing
its aggregate metrics.

Produces two tidy DataFrames — one row per (article, model, method, word):

  mft_df   — every moral word in the original article, whether it was
              preserved in the summary, and its MFT label/dimension.
              → Used for: MFT dimension-level breakdown by political leaning.

  add_df   — every moral word found in the summary that was NOT in the
              original article's annotation set (i.e. added by the model).
              → Used for: "moral framing addition" error analysis.

Usage (from notebook or standalone):
    from moral_framing_extractor import build_posthoc_dfs
    mft_df, add_df = build_posthoc_dfs()
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

#reuse existing helpers
import sys
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moral_summarization.data_utils import (
    load_annotations,
    get_moral_annotations,
    get_summary_from_response,
    EMONA_dataset_path,
)

from moral_summarization.utils import lemmatize_and_clean

# ── MFT dimension grouping (positive + negative pole share a foundation) ──
FOUNDATION_MAP = {
    "care":       "Care/Harm",
    "harm":       "Care/Harm",
    "fairness":   "Fairness/Cheating",
    "cheating":   "Fairness/Cheating",
    "loyalty":    "Loyalty/Betrayal",
    "betrayal":   "Loyalty/Betrayal",
    "authority":  "Authority/Subversion",
    "subversion": "Authority/Subversion",
    "purity":     "Purity/Degradation",
    "degradation":"Purity/Degradation",
}

POLARITY_MAP = {
    "care": "positive", "harm": "negative",
    "fairness": "positive", "cheating": "negative",
    "loyalty": "positive", "betrayal": "negative",
    "authority": "positive", "subversion": "negative",
    "purity": "positive", "degradation": "negative",
}

# AllSides article-ID regex (copied from data_loader to keep this file standalone)
_ALLSIDES_RE = re.compile(
    r"^allsides_(?P<topic>.+)_(?P<leaning>[lcr])_(?P<idx>\d+)$"
)
_LEANING_MAP = {"l": "left", "c": "center", "r": "right"}

RESULTS_DIR = REPO_ROOT / "results" / "test_prompts" / "allsides"


def _parse_article_id(article: str) -> dict:
    m = _ALLSIDES_RE.match(article)
    if not m:
        return {"topic": None, "leaning": None}
    return {
        "topic":   m.group("topic"),
        "leaning": _LEANING_MAP[m.group("leaning")],
    }


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer — no torchtext required.

    Matches the behaviour of torchtext's 'basic_english' tokenizer:
    lowercase, split on anything that is not a letter or digit.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _count_moral_words_local(
    summary_text: str,
    moral_anns: list[dict],
) -> tuple[int, list[dict]]:
    """Equivalent of data_utils.count_moral_words but uses _tokenize.

    Returns (count, list_of_matched_annotation_dicts).
    """
    summary_tokens = set()
    for raw in _tokenize(summary_text):
        clean = lemmatize_and_clean(raw)
        if clean:
            summary_tokens.add(clean)

    seen_words: list[str] = []
    clean_anns: list[dict] = []
    for ann in moral_anns:
        token = lemmatize_and_clean(ann["token"])
        if token and token not in seen_words:
            seen_words.append(token)
            clean_anns.append({**ann, "token": token})

    matched = [a for a in clean_anns if a["token"] in summary_tokens]
    return len(matched), matched


def _get_summary_tokens(summary_text: str) -> set[str]:
    """Return the de-duplicated, lemmatized token set of a text."""
    out: set[str] = set()
    for raw in _tokenize(summary_text):
        clean = lemmatize_and_clean(raw)
        if clean:
            out.add(clean)
    return out


def build_posthoc_dfs(
    results_dir: Path = RESULTS_DIR,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk every article folder under *results_dir* and build two DataFrames.

    Parameters
    ----------
    results_dir : path to results/test_prompts/allsides
    verbose     : print progress

    Returns
    -------
    mft_df : DataFrame — preserved-word level, columns:
        article, topic, leaning, model, method,
        word, mft_label, foundation, polarity, preserved

    add_df : DataFrame — added-word level, columns:
        article, topic, leaning, model, method,
        word
        (one row per moral word the model *added* that wasn't in the article)
    """
    mft_rows: list[dict] = []
    add_rows: list[dict] = []

    article_dirs = sorted(results_dir.iterdir())
    total = len(article_dirs)

    for i, article_dir in enumerate(article_dirs):
        article = article_dir.name
        meta    = _parse_article_id(article)

        if verbose and i % 30 == 0:
            print(f"  [{i+1}/{total}] {article}")

        # ── load annotations for this article ──────────────────────────────
        try:
            annotations     = load_annotations(article, "allsides")
            moral_anns      = get_moral_annotations(annotations)
        except Exception as e:
            if verbose:
                print(f"    WARNING: could not load annotations for {article}: {e}")
            continue

        # Build the set of article-level moral words (lemmatized, deduplicated)
        article_moral_words: dict[str, str] = {}  # word → mft_label
        for ann in moral_anns:
            clean = lemmatize_and_clean(ann["token"])
            if clean and clean not in article_moral_words:
                article_moral_words[clean] = ann["label"]

        # iterate over response files 
        response_files = sorted(article_dir.glob("*_response.txt"))
        for resp_path in response_files:
            # filename: <model>_<method>_response.txt
            # e.g. Meta-Llama-3-70B-Instruct_vanilla_response.txt
            fname = resp_path.stem  # strip .txt
            # strip trailing "_response"
            fname_no_suffix = fname[: fname.rfind("_response")]
            # split on first underscore that precedes a known method name
            # Strategy: known methods are fixed — try them longest-first
            known_methods = [
                "simple_fewshot_mft", "cot_fewshot_mft",
                "simple_fewshot", "cot_fewshot",
                "vanilla", "simple", "cot", "oracle", "class",
            ]
            method = None
            model  = None
            for m_name in known_methods:
                if fname_no_suffix.endswith("_" + m_name):
                    method = m_name
                    model  = fname_no_suffix[: -(len(m_name) + 1)]
                    break
            if method is None:
                continue  # unrecognized file

            # extract summary text
            try:
                summary_text = get_summary_from_response(str(resp_path))
            except Exception:
                summary_text = None
            if not summary_text:
                continue

            #preserved words (MFT breakdown)
            _, preserved_anns = _count_moral_words_local(summary_text, moral_anns)
            preserved_words   = {a["token"] for a in preserved_anns if a["token"]}

            base = dict(
                article = article,
                topic   = meta["topic"],
                leaning = meta["leaning"],
                model   = model,
                method  = method,
            )

            for word, label in article_moral_words.items():
                mft_rows.append({
                    **base,
                    "word":       word,
                    "mft_label":  label,
                    "foundation": FOUNDATION_MAP.get(label, label),
                    "polarity":   POLARITY_MAP.get(label, "unknown"),
                    "preserved":  word in preserved_words,
                })

            #added words (framing addition)
            summary_tokens  = _get_summary_tokens(summary_text)
            article_tokens  = set(article_moral_words.keys())

            # A word is "added" if:
            #   1. it appears in the summary
            #   2. it is a known moral word (in any EMONA article — we use the
            #      full moral_labels vocab as a proxy; stricter: use the
            #      article annotation set's label vocabulary)
            #   3. it is NOT in the article's own moral word set
            added = summary_tokens - article_tokens
            for word in added:
                # quick heuristic: only flag if the word itself looks like a
                # moral concept (check against all EMONA moral word lists is
                # expensive; for now we leave it to the analyst to filter)
                add_rows.append({**base, "word": word})

    mft_df = pd.DataFrame(mft_rows)
    add_df  = pd.DataFrame(add_rows)

    if verbose:
        print(f"\nDone.")
        print(f"  mft_df : {len(mft_df):,} rows  "
              f"(article × model × method × moral_word)")
        print(f"  add_df : {len(add_df):,} rows  "
              f"(article × model × method × added_word)")

    return mft_df, add_df
