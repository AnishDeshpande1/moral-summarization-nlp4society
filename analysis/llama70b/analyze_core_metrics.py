"""Analyze the core evaluation metrics for the Meta-Llama-3-70B-Instruct run.

Read-only over the evaluation pickle. Writes flat into analysis/gptoss/.

Metrics (all relative to EMONA human moral-word annotations):
  moral_count : # of the article's annotated moral words that survive into the
                summary (higher = more moral content preserved)
  moral_div   : Jensen-Shannon divergence between the moral-foundation
                distribution of the article vs. the summary (LOWER = better)
  length      : summary length in tokens (context only)

Usage:
  python analysis/gptoss/analyze_core_metrics.py
"""
import os
import pickle
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PICKLE = os.path.join(HERE, 'core_metrics.pickle')
OUT_DIR = HERE

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot']
METRICS = ['moral_count', 'moral_div', 'length']


def load_metrics():
    with open(PICKLE, 'rb') as f:
        return pickle.load(f)


def build_summary_table(metrics):
    rows = []
    for metric in METRICS:
        df = metrics[metric]
        mean_rows = df[df.index.get_level_values('dataset') == 'mean'].droplevel('model')
        for dataset in mean_rows.index.get_level_values('article').unique():
            sub = mean_rows[mean_rows.index.get_level_values('article') == dataset]
            for strat in STRATEGIES:
                rows.append({
                    'metric': metric, 'dataset': dataset, 'strategy': strat,
                    'mean': round(float(sub[strat].iloc[0]), 3),
                })
    return pd.DataFrame(rows)


def build_long_form(metrics):
    frames = []
    for metric in METRICS:
        df = metrics[metric].droplevel('model')
        df = df[~df.index.get_level_values('dataset').isin(['mean', 'std'])]
        long = df[STRATEGIES].reset_index().melt(
            id_vars=['dataset', 'article'], value_vars=STRATEGIES,
            var_name='strategy', value_name='value')
        long['metric'] = metric
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def coverage_counts(long_df):
    mc = long_df[long_df['metric'] == 'moral_count']
    total = mc.groupby('strategy')['article'].nunique()
    usable = mc.dropna(subset=['value']).groupby('strategy')['article'].nunique()
    out = pd.DataFrame({'articles_total': total, 'articles_with_summary': usable})
    out['articles_with_summary'] = out['articles_with_summary'].fillna(0).astype(int)
    out['parse_failure_rate'] = (
        1 - out['articles_with_summary'] / out['articles_total']).round(3)
    return out.reindex(STRATEGIES)


def print_insights(summary, coverage):
    print('=' * 78)
    print('CORE METRIC INSIGHTS  (model: Meta-Llama-3-70B-Instruct, 400 articles)')
    print('=' * 78)
    allrows = summary[summary['dataset'] == 'all'].set_index(['metric', 'strategy'])['mean']

    print('\n--- moral_count: moral words preserved (higher = better) ---')
    mc = allrows['moral_count'].reindex(STRATEGIES)
    print(mc.to_string())
    best = mc.drop('oracle').idxmax()
    print(f"  > best non-oracle: {best} ({mc[best]:.2f}); "
          f"oracle ceiling: {mc['oracle']:.2f}; vanilla: {mc['vanilla']:.2f}")

    print('\n--- moral_div: frame divergence (LOWER = better) ---')
    md = allrows['moral_div'].reindex(STRATEGIES)
    print(md.to_string())
    print(f"  > best non-oracle (lowest): {md.drop('oracle').idxmin()} "
          f"({md.drop('oracle').min():.3f})")

    print('\n--- length ---')
    print(allrows['length'].reindex(STRATEGIES).to_string())

    print('\n--- zero-shot vs few-shot ---')
    for metric, better in [('moral_count', 'higher'), ('moral_div', 'lower')]:
        z = allrows[metric].reindex(ZEROSHOT).mean()
        f = allrows[metric].reindex(FEWSHOT).mean()
        verdict = 'few-shot better' if (
            (better == 'higher' and f > z) or (better == 'lower' and f < z)
        ) else 'zero-shot better'
        print(f"  {metric}: zeroshot={z:.3f} fewshot={f:.3f} -> {verdict}")

    print('\n--- summary parse coverage ---')
    print(coverage.to_string())


def main():
    metrics = load_metrics()
    summary = build_summary_table(metrics)
    long_df = build_long_form(metrics)
    coverage = coverage_counts(long_df)

    summary.to_csv(os.path.join(OUT_DIR, 'strategy_summary.csv'), index=False)
    long_df.to_csv(os.path.join(OUT_DIR, 'per_article_long.csv'), index=False)
    coverage.to_csv(os.path.join(OUT_DIR, 'parse_coverage.csv'))
    print_insights(summary, coverage)
    print(f"\nWrote strategy_summary.csv, per_article_long.csv, parse_coverage.csv -> {OUT_DIR}")


if __name__ == '__main__':
    main()
