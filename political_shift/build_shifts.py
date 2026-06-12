"""
Build ``shifts.parquet`` -- one row per ``(article_id, method, model, seed)`` joining
the article-side stance with the summary-side stance.

Output columns:
    article_id, method, model, seed, leaning, topic,
    article_direction, summary_direction, direction_shift,
    article_polarization, summary_polarization, polarization_shift,
    p_left_art, p_center_art, p_right_art,
    p_left_sum, p_center_sum, p_right_sum.

Both scalar scores are derived directly from the cached raw probabilities so that
this script can be re-run without re-running BERT:
    direction    = P(right) - P(left)   in [-1, +1];  +1 = right-leaning
    polarization = 1 - P(center)        in [0, 1];     0 = fully centrist
    *_shift      = summary_* - article_*

Run from the repo root:

    python -m political_shift.build_shifts
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "political_shift" / "data"
TEXTS_PATH = DATA_DIR / "texts.parquet"
SCORES_PATH = DATA_DIR / "stance_scores.parquet"
OUT_PATH = DATA_DIR / "shifts.parquet"


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
def build_shifts() -> pd.DataFrame:
    """Join texts + stance scores into the (article, summary) shift table."""
    texts = pd.read_parquet(TEXTS_PATH)
    scores = pd.read_parquet(SCORES_PATH)

    merged = texts.merge(scores, on="text_id", how="inner")

    # Direction is recomputed from raw probs (P(right) - P(left)) so this script
    # produces correct signs without needing to re-run BERT scoring.
    articles = (
        merged[merged["kind"] == "article"]
        [["article_id", "leaning", "topic", "p_left", "p_center", "p_right", "polarization"]]
        .copy()
        .rename(columns={
            "p_left": "p_left_art",
            "p_center": "p_center_art",
            "p_right": "p_right_art",
            "polarization": "article_polarization",
        })
        .assign(article_direction=lambda df: df["p_right_art"] - df["p_left_art"])
    )

    summaries = (
        merged[merged["kind"] == "summary"]
        [["article_id", "method", "model", "seed",
          "p_left", "p_center", "p_right", "polarization"]]
        .copy()
        .rename(columns={
            "p_left": "p_left_sum",
            "p_center": "p_center_sum",
            "p_right": "p_right_sum",
            "polarization": "summary_polarization",
        })
        .assign(summary_direction=lambda df: df["p_right_sum"] - df["p_left_sum"])
    )

    shifts = summaries.merge(articles, on="article_id", how="inner")
    shifts["direction_shift"] = shifts["summary_direction"] - shifts["article_direction"]
    shifts["polarization_shift"] = shifts["summary_polarization"] - shifts["article_polarization"]

    col_order = [
        "article_id", "method", "model", "seed",
        "leaning", "topic",
        "article_direction", "summary_direction", "direction_shift",
        "article_polarization", "summary_polarization", "polarization_shift",
        "p_left_art", "p_center_art", "p_right_art",
        "p_left_sum", "p_center_sum", "p_right_sum",
    ]
    return shifts[col_order].sort_values(
        ["model", "method", "leaning", "article_id", "seed"]
    ).reset_index(drop=True)


def main() -> None:
    if not TEXTS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEXTS_PATH}. Run build_texts.py first.")
    if not SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing {SCORES_PATH}. Run score_texts.py first.")

    shifts = build_shifts()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shifts.to_parquet(OUT_PATH, index=False)

    n_scored = len(shifts) + shifts["article_id"].nunique()  # summaries + their articles
    mean_dir = shifts.groupby("leaning")["direction_shift"].mean().round(4).to_dict()
    mean_pol = shifts.groupby("leaning")["polarization_shift"].mean().round(4).to_dict()

    print(f"Wrote {len(shifts):,} rows to {OUT_PATH}")
    print(f"Total texts scored (summaries + unique articles): {n_scored:,}")
    print("Mean direction_shift per leaning (across all models/methods):")
    for leaning in ("left", "center", "right"):
        if leaning in mean_dir:
            print(f"  {leaning:<6}: {mean_dir[leaning]:+.4f}")
    print("Mean polarization_shift per leaning (across all models/methods):")
    for leaning in ("left", "center", "right"):
        if leaning in mean_pol:
            print(f"  {leaning:<6}: {mean_pol[leaning]:+.4f}")

    # Sanity: with direction = P(right) - P(left), left-leaning articles should
    # score negative and right-leaning articles should score positive.
    art_dir = shifts.groupby("leaning")["article_direction"].mean().round(4)
    print("\nSanity — mean article_direction by leaning  (expect left < 0, right > 0):")
    for lean in ("left", "center", "right"):
        val = art_dir.get(lean, float("nan"))
        ok = (lean == "left" and val < 0) or (lean == "right" and val > 0) or lean == "center"
        print(f"  {lean:<6}: {val:+.4f}  {'✓' if ok else '✗ UNEXPECTED'}")

    print(f"\nFinal parquet: {OUT_PATH}")


if __name__ == "__main__":
    main()
