"""Paired (intersection) comparison for the Llama-3.1-8B run.

The plain per-strategy means in analyze_core_metrics.py are each averaged over
whichever articles that strategy successfully summarized. Different strategies
refuse / fail to parse on different articles, so those means are computed over
DIFFERENT article sets and are not directly comparable.

This script restricts to the COMMON set of articles where every compared pair
of strategies produced a usable summary (non-NaN metric), then:
  - recomputes per-strategy means on that common set
  - runs paired tests (Wilcoxon signed-rank + paired t-test) for each contrast

Read-only. Writes flat into analysis/data/.

Usage:
  python analysis/paired_comparison.py
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'llama')
LONG_CSV = os.path.join(OUT_DIR, 'per_article_long.csv')

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot']
METRIC_DIRECTION = {'moral_count': 'higher', 'moral_div': 'lower'}


def load_pivot(metric):
    df = pd.read_csv(LONG_CSV)
    m = df[df['metric'] == metric].copy()
    m['key'] = m['dataset'] + '/' + m['article']
    return m.pivot(index='key', columns='strategy', values='value')


def common_subset(pivot, strategies):
    return pivot.dropna(subset=strategies)


def paired_means(metric, strategies):
    piv = load_pivot(metric)
    common = common_subset(piv, strategies)
    means = common[strategies].mean().round(3)
    return means, len(common)


def paired_test(metric, strat_a, strat_b):
    piv = load_pivot(metric)
    common = common_subset(piv, [strat_a, strat_b])
    a = common[strat_a].values
    b = common[strat_b].values
    n = len(common)

    diff = a.mean() - b.mean()
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
    piv = load_pivot(metric)
    common = common_subset(piv, common_strats)
    return common[group].mean().mean()


def main():
    rows = []
    print('=' * 78)
    print('PAIRED (INTERSECTION) COMPARISON  -  Llama-3.1-8B')
    print('=' * 78)
    print('Means are computed only on articles where BOTH compared strategies'
          '\nproduced a usable summary, so the comparison is apples-to-apples.')

    # per-strategy means on the all-non-oracle common set
    non_oracle = [s for s in STRATEGIES if s != 'oracle']
    print('\n--- per-strategy means on the common set (all non-oracle strategies) ---')
    for metric in ['moral_count', 'moral_div']:
        means, n = paired_means(metric, non_oracle)
        print(f"\n  {metric}  (n_common = {n}, {METRIC_DIRECTION[metric]} better)")
        print(means.reindex(non_oracle).to_string())

    # zero-shot vs few-shot group means
    print('\n--- zero-shot vs few-shot (group means on common set) ---')
    for metric in ['moral_count', 'moral_div']:
        common_strats = ZEROSHOT + FEWSHOT
        zs = group_mean_on_common(metric, ZEROSHOT, common_strats)
        fs = group_mean_on_common(metric, FEWSHOT, common_strats)
        d = METRIC_DIRECTION[metric]
        better = 'few-shot' if ((d == 'higher' and fs > zs) or (d == 'lower' and fs < zs)) else 'zero-shot'
        print(f"  {metric}: zeroshot={zs:.3f}  fewshot={fs:.3f}  -> {better} better")

    # pairwise paired tests
    print('\n--- pairwise paired tests (Wilcoxon + paired t) ---')
    contrasts = [
        # few-shot vs matched zero-shot baseline
        ('moral_count', 'cot_fewshot',        'cot'),
        ('moral_count', 'cot_fewshot_mft',    'cot'),
        ('moral_count', 'simple_fewshot',     'simple'),
        ('moral_count', 'simple_fewshot_mft', 'simple'),
        # zero-shot cot vs simple
        ('moral_count', 'cot',                'simple'),
        # few-shot cot vs few-shot simple
        ('moral_count', 'cot_fewshot',        'simple_fewshot'),
        ('moral_count', 'cot_fewshot_mft',    'simple_fewshot_mft'),
        # same for moral_div
        ('moral_div',   'cot_fewshot',        'cot'),
        ('moral_div',   'cot_fewshot_mft',    'cot'),
        ('moral_div',   'simple_fewshot',     'simple'),
        ('moral_div',   'simple_fewshot_mft', 'simple'),
        ('moral_div',   'cot',                'simple'),
        ('moral_div',   'cot_fewshot',        'simple_fewshot'),
        ('moral_div',   'cot_fewshot_mft',    'simple_fewshot_mft'),
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
