"""Paired (intersection) comparison for the DeepSeek-R1-Distill-Qwen-32B run.

The plain per-strategy means in analyze_core_metrics.py are each averaged over
whichever articles that strategy successfully summarized. Different strategies
refuse / fail to parse on different articles, so those means are computed over
DIFFERENT article sets and are not directly comparable.

This script restricts to the COMMON set of articles where every strategy under
comparison produced a usable summary (non-NaN metric), then:
  - recomputes per-strategy means on that common set
  - runs paired tests (Wilcoxon signed-rank + paired t-test) for the key
    contrasts, since every strategy is now scored on the same articles.

Read-only. Writes flat into analysis/gptoss/.

Usage:
  python analysis/gptoss/paired_comparison.py
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE
LONG_CSV = os.path.join(HERE, 'per_article_long.csv')

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot']  # oracle excluded: it's a ceiling, refuses often
# "higher is better" for moral_count, "lower is better" for moral_div
METRIC_DIRECTION = {'moral_count': 'higher', 'moral_div': 'lower'}


def load_pivot(metric):
    df = pd.read_csv(LONG_CSV)
    m = df[df['metric'] == metric].copy()
    m['key'] = m['dataset'] + '/' + m['article']
    return m.pivot(index='key', columns='strategy', values='value')


def common_subset(pivot, strategies):
    """Rows where ALL listed strategies have a non-NaN value."""
    return pivot.dropna(subset=strategies)


def paired_means(metric, strategies):
    """Per-strategy mean on the common (intersection) article set."""
    piv = load_pivot(metric)
    common = common_subset(piv, strategies)
    means = common[strategies].mean().round(3)
    return means, len(common)


def paired_test(metric, strat_a, strat_b):
    """Paired Wilcoxon + t-test between two strategies on their common articles.

    Returns dict with means, difference, and p-values. 'better' is reported per
    the metric direction.
    """
    piv = load_pivot(metric)
    common = common_subset(piv, [strat_a, strat_b])
    a = common[strat_a].values
    b = common[strat_b].values
    n = len(common)

    diff = a.mean() - b.mean()
    # Wilcoxon needs some non-zero differences
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    t_stat, t_p = ttest_rel(a, b)

    direction = METRIC_DIRECTION[metric]
    if direction == 'higher':
        winner = strat_a if a.mean() > b.mean() else strat_b
    else:
        winner = strat_a if a.mean() < b.mean() else strat_b

    return {
        'metric': metric, 'strat_a': strat_a, 'strat_b': strat_b, 'n_common': n,
        'mean_a': round(a.mean(), 3), 'mean_b': round(b.mean(), 3),
        'mean_diff_a_minus_b': round(diff, 3),
        'wilcoxon_p': round(w_p, 5) if not np.isnan(w_p) else np.nan,
        'ttest_p': round(t_p, 5),
        'better': winner, 'direction': direction,
    }


def group_mean_on_common(metric, group, common_strats):
    """Mean of a strategy GROUP, computed on the set common to common_strats."""
    piv = load_pivot(metric)
    common = common_subset(piv, common_strats)
    return common[group].mean().mean()


def main():
    rows = []
    print('=' * 78)
    print('PAIRED (INTERSECTION) COMPARISON  -  DeepSeek-R1-Distill-Qwen-32B')
    print('=' * 78)
    print('Means are computed only on articles where BOTH/ALL compared strategies'
          '\nproduced a usable summary, so the comparison is apples-to-apples.')

    # 1. Per-strategy means on the all-non-oracle common set (consistent denominator)
    non_oracle = [s for s in STRATEGIES if s != 'oracle']
    print('\n--- per-strategy means on the common set (all non-oracle strategies) ---')
    for metric in ['moral_count', 'moral_div']:
        means, n = paired_means(metric, non_oracle)
        print(f"\n  {metric}  (n_common = {n}, {METRIC_DIRECTION[metric]} better)")
        print(means.reindex(non_oracle).to_string())

    # 2. zero-shot vs few-shot, group means on the common set + paired test on best of each
    print('\n--- zero-shot vs few-shot (group means on common set) ---')
    for metric in ['moral_count', 'moral_div']:
        common_strats = ZEROSHOT + FEWSHOT
        zs = group_mean_on_common(metric, ZEROSHOT, common_strats)
        fs = group_mean_on_common(metric, FEWSHOT, common_strats)
        d = METRIC_DIRECTION[metric]
        better = 'few-shot' if ((d == 'higher' and fs > zs) or (d == 'lower' and fs < zs)) else 'zero-shot'
        print(f"  {metric}: zeroshot={zs:.3f}  fewshot={fs:.3f}  -> {better} better")

    # 3. Key pairwise paired tests
    print('\n--- key pairwise paired tests (Wilcoxon + paired t) ---')
    contrasts = [
        # few-shot vs matched zero-shot baseline
        ('moral_count', 'cot_fewshot',     'cot'),
        ('moral_count', 'cot_fewshot_mft', 'cot'),
        ('moral_count', 'simple_fewshot',  'simple'),
        ('moral_count', 'simple_fewshot_mft', 'simple'),
        # zero-shot cot vs simple
        ('moral_count', 'cot',             'simple'),
        # few-shot cot vs few-shot simple (same reasoning style, different base)
        ('moral_count', 'cot_fewshot',     'simple_fewshot'),
        ('moral_count', 'cot_fewshot_mft', 'simple_fewshot_mft'),
        # same 7 contrasts for moral_div
        ('moral_div',   'cot_fewshot',     'cot'),
        ('moral_div',   'cot_fewshot_mft', 'cot'),
        ('moral_div',   'simple_fewshot',  'simple'),
        ('moral_div',   'simple_fewshot_mft', 'simple'),
        ('moral_div',   'cot',             'simple'),
        ('moral_div',   'cot_fewshot',     'simple_fewshot'),
        ('moral_div',   'cot_fewshot_mft', 'simple_fewshot_mft'),
    ]
    for metric, a, b in contrasts:
        r = paired_test(metric, a, b)
        rows.append(r)
        sig = ''
        if not np.isnan(r['wilcoxon_p']):
            sig = '***' if r['wilcoxon_p'] < 0.001 else '**' if r['wilcoxon_p'] < 0.01 else '*' if r['wilcoxon_p'] < 0.05 else 'ns'
        print(f"  [{metric:11}] {a} vs {b}: "
              f"means {r['mean_a']} vs {r['mean_b']} (n={r['n_common']}), "
              f"Wilcoxon p={r['wilcoxon_p']} {sig}, better={r['better']}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'paired_comparison.csv')
    out.to_csv(out_path, index=False)
    print(f"\nWrote paired_comparison.csv -> {OUT_DIR}")
    print('Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant')


if __name__ == '__main__':
    main()
