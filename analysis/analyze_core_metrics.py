"""Analyze the core evaluation metrics produced by run_evaluation.py.

Read-only over the evaluation pickle. Produces:
  - analysis/data/strategy_summary.csv  (mean per strategy x metric x dataset)
  - analysis/data/per_article_long.csv  (tidy long-form, one row per article/strategy/metric)
  - printed insights

Metrics recap (all relative to EMONA human moral-word annotations):
  moral_count : # of the article's annotated moral words that survive into the summary
                (higher = more moral content preserved)
  moral_div   : Jensen-Shannon divergence between the moral-foundation distribution of
                the article vs. the summary (LOWER = better moral-frame preservation)
  length      : summary length in tokens (context, not a quality metric on its own)

Usage:
  python analysis/analyze_core_metrics.py
"""
import os
import pickle
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'analysis', 'llama')
PICKLE = os.path.join(HERE, 'core_metrics.pickle')
OUT_DIR = HERE

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
# Strategies that are part of *our* few-shot contribution (vs. the paper's originals).
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot']  # oracle is a ceiling, treated separately
METRICS = ['moral_count', 'moral_div', 'length']


def load_metrics():
    with open(PICKLE, 'rb') as f:
        return pickle.load(f)


def build_summary_table(metrics):
    """Mean per strategy, per dataset (+ 'all'), one block per metric."""
    rows = []
    for metric in METRICS:
        df = metrics[metric]
        mean_rows = df[df.index.get_level_values('dataset') == 'mean'].droplevel('model')
        for dataset in mean_rows.index.get_level_values('article').unique():
            sub = mean_rows[mean_rows.index.get_level_values('article') == dataset]
            for strat in STRATEGIES:
                rows.append({
                    'metric': metric,
                    'dataset': dataset,
                    'strategy': strat,
                    'mean': round(float(sub[strat].iloc[0]), 3),
                })
    return pd.DataFrame(rows)


def build_long_form(metrics):
    """Tidy per-article values (drop the appended mean/std aggregate rows)."""
    frames = []
    for metric in METRICS:
        df = metrics[metric].droplevel('model')
        df = df[~df.index.get_level_values('dataset').isin(['mean', 'std'])]
        long = df[STRATEGIES].reset_index().melt(
            id_vars=['dataset', 'article'],
            value_vars=STRATEGIES,
            var_name='strategy', value_name='value',
        )
        long['metric'] = metric
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def coverage_counts(long_df):
    """How many articles produced a usable summary per strategy (non-NaN moral_count)."""
    mc = long_df[long_df['metric'] == 'moral_count']
    total = mc.groupby('strategy')['article'].nunique()
    usable = mc.dropna(subset=['value']).groupby('strategy')['article'].nunique()
    out = pd.DataFrame({'articles_total': total, 'articles_with_summary': usable})
    out['articles_with_summary'] = out['articles_with_summary'].fillna(0).astype(int)
    out['parse_failure_rate'] = (
        1 - out['articles_with_summary'] / out['articles_total']
    ).round(3)
    return out.reindex(STRATEGIES)


def print_insights(summary, coverage):
    print('=' * 78)
    print('CORE METRIC INSIGHTS  (model: llama3.1-8b-instruct-q4_K_M, 400 articles)')
    print('=' * 78)

    allrows = summary[summary['dataset'] == 'all'].set_index(['metric', 'strategy'])['mean']

    print('\n--- moral_count: moral words preserved (higher = better) ---')
    mc = allrows['moral_count'].reindex(STRATEGIES)
    print(mc.to_string())
    best_real = mc.drop('oracle').idxmax()
    print(f"  > Best non-oracle strategy: {best_real} ({mc[best_real]:.2f})")
    print(f"  > Oracle (ceiling, injects ground-truth words): {mc['oracle']:.2f}")
    print(f"  > vanilla baseline: {mc['vanilla']:.2f} | "
          f"best few-shot lifts it by {mc[FEWSHOT].max() - mc['vanilla']:+.2f}")

    print('\n--- moral_div: JS divergence of moral-frame distribution (LOWER = better) ---')
    md = allrows['moral_div'].reindex(STRATEGIES)
    print(md.to_string())
    best_div = md.drop('oracle').idxmin()
    print(f"  > Best non-oracle (lowest divergence): {best_div} ({md[best_div]:.3f})")
    print(f"  > Oracle: {md['oracle']:.3f} (lowest, as expected)")

    print('\n--- length: summary tokens (context) ---')
    ln = allrows['length'].reindex(STRATEGIES)
    print(ln.to_string())

    print('\n--- zero-shot vs few-shot (mean over the metric, across datasets) ---')
    for metric, better in [('moral_count', 'higher'), ('moral_div', 'lower')]:
        z = allrows[metric].reindex(ZEROSHOT).mean()
        f = allrows[metric].reindex(FEWSHOT).mean()
        arrow = '>' if better == 'higher' else '<'
        verdict = 'few-shot better' if (
            (better == 'higher' and f > z) or (better == 'lower' and f < z)
        ) else 'zero-shot better'
        print(f"  {metric}: zeroshot={z:.3f}  fewshot={f:.3f}  "
              f"(want fewshot {arrow} zeroshot) -> {verdict}")

    print('\n--- summary parse coverage (non-refusal, well-formed SUMMARY block) ---')
    print(coverage.to_string())
    worst = coverage['parse_failure_rate'].idxmax()
    print(f"  > Highest parse-failure strategy: {worst} "
          f"({coverage.loc[worst, 'parse_failure_rate']:.1%})")
    print('  (parse failures = refusals or missing SUMMARY/END tokens; '
          'see refusal_analysis.py for the breakdown)')


def main():
    metrics = load_metrics()
    summary = build_summary_table(metrics)
    long_df = build_long_form(metrics)
    coverage = coverage_counts(long_df)

    summary.to_csv(os.path.join(OUT_DIR, 'strategy_summary.csv'), index=False)
    long_df.to_csv(os.path.join(OUT_DIR, 'per_article_long.csv'), index=False)
    coverage.to_csv(os.path.join(OUT_DIR, 'parse_coverage.csv'))

    print_insights(summary, coverage)
    print(f"\nWrote: strategy_summary.csv, per_article_long.csv, parse_coverage.csv "
          f"-> {OUT_DIR}")


if __name__ == '__main__':
    main()
