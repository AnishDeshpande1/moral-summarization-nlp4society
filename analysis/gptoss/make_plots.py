"""Generate figures for the GPT-OSS-120B evaluation.

Read-only: consumes the CSVs produced by the other two scripts in this folder.
Writes PNGs to analysis/gptoss/figures/.

Usage:
  python analysis/gptoss/make_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE
FIGS = os.path.join(HERE, 'figures')
os.makedirs(FIGS, exist_ok=True)

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = {'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft'}
LABELS = {
    'vanilla': 'vanilla', 'simple': 'simple', 'cot': 'cot', 'oracle': 'oracle\n(ceiling)',
    'simple_fewshot': 'simple\nfewshot', 'cot_fewshot': 'cot\nfewshot',
    'simple_fewshot_mft': 'simple\nfewshot+mft', 'cot_fewshot_mft': 'cot\nfewshot+mft',
}


def bar_colors(strats):
    out = []
    for s in strats:
        if s == 'oracle':
            out.append('#d4a017')
        elif s in FEWSHOT:
            out.append('#2a9d8f')
        else:
            out.append('#8d99ae')
    return out


LEGEND = [
    Patch(facecolor='#8d99ae', label='Zero-shot (baseline)'),
    Patch(facecolor='#d4a017', label='Oracle (ceiling)'),
    Patch(facecolor='#2a9d8f', label='Few-shot (our contribution)'),
]
TITLE = 'GPT-OSS-120B'


def _all_mean(summary, metric):
    sub = summary[(summary['metric'] == metric) & (summary['dataset'] == 'all')]
    return sub.set_index('strategy')['mean'].reindex(STRATEGIES)


def fig_moral_count(summary):
    vals = _all_mean(summary, 'moral_count')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(STRATEGIES)), vals.values, color=bar_colors(STRATEGIES))
    best = vals.drop('oracle').idxmax()
    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.12, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel('Moral words preserved (mean over 400 articles)')
    ax.set_title(f'{TITLE}: moral-word preservation  (higher = better)\n'
                 f'best non-oracle: {best} ({vals[best]:.2f})', fontsize=12)
    ax.legend(handles=LEGEND, fontsize=8, loc='upper left')
    ax.set_ylim(0, vals.max() * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'moral_count_by_strategy.png'), dpi=150)
    plt.close(fig)


def fig_moral_div(summary):
    vals = _all_mean(summary, 'moral_div')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(STRATEGIES)), vals.values, color=bar_colors(STRATEGIES))
    best = vals.drop('oracle').idxmin()
    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel('JS divergence of moral-frame distribution')
    ax.set_title(f'{TITLE}: moral-frame divergence  (LOWER = better)\n'
                 f'best non-oracle: {best} ({vals[best]:.3f})', fontsize=12)
    ax.legend(handles=LEGEND, fontsize=8, loc='upper right')
    ax.set_ylim(0, vals.max() * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'moral_div_by_strategy.png'), dpi=150)
    plt.close(fig)


def fig_refusal_rate(refusal):
    r = refusal.set_index('strategy').reindex(STRATEGIES)
    rates = r['refusal_rate'] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(STRATEGIES)), rates.values, color=bar_colors(STRATEGIES))
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel('Refusal rate (% of 400 articles)')
    ax.set_title(f'{TITLE}: refusals by strategy  (lower = better)', fontsize=12)
    ax.legend(handles=LEGEND, fontsize=8, loc='upper right')
    ax.set_ylim(0, max(rates.max() * 1.2, 5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'refusal_rate_by_strategy.png'), dpi=150)
    plt.close(fig)


def fig_refusal_subtype(refusal):
    """Stacked: safety vs moderation refusals per strategy (GPT-OSS specific)."""
    r = refusal.set_index('strategy').reindex(STRATEGIES)
    safety = r['refusal_safety'].values
    moderation = r['refusal_moderation'].values
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(STRATEGIES)), safety, color='#e76f51', label='safety (model declines)')
    ax.bar(range(len(STRATEGIES)), moderation, bottom=safety,
           color='#6a4c93', label='moderation (provider filter)')
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel('Refusals (of 400)')
    ax.set_title(f'{TITLE}: refusal source by strategy', fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'refusal_subtype_by_strategy.png'), dpi=150)
    plt.close(fig)


def fig_response_breakdown(refusal):
    r = refusal.set_index('strategy').reindex(STRATEGIES)
    cats = ['ok', 'format_miss', 'refusal', 'no_summary']
    colors = {'ok': '#2a9d8f', 'format_miss': '#e9c46a',
              'refusal': '#e76f51', 'no_summary': '#a8a8a8'}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(STRATEGIES))
    for cat in cats:
        vals = r[cat].values
        ax.bar(range(len(STRATEGIES)), vals, bottom=bottom, color=colors[cat], label=cat)
        bottom += vals
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel('Responses (of 400)')
    ax.set_title(f'{TITLE}: response outcome breakdown by strategy', fontsize=12)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.set_ylim(0, 420)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'response_breakdown_stacked.png'), dpi=150)
    plt.close(fig)


def fig_zeroshot_vs_fewshot(summary, refusal):
    zs = ['vanilla', 'simple', 'cot', 'oracle']
    fs = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
    mc = _all_mean(summary, 'moral_count')
    md = _all_mean(summary, 'moral_div')
    r = refusal.set_index('strategy')['refusal_rate']
    groups = {'Zero-shot': zs, 'Few-shot': fs}
    metrics = {
        'Moral words\n(higher better)': lambda g: mc.reindex(g).mean(),
        'Frame divergence\n(lower better)': lambda g: md.reindex(g).mean(),
        'Refusal rate %\n(lower better)': lambda g: r.reindex(g).mean() * 100,
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (name, fn) in zip(axes, metrics.items()):
        vals = [fn(groups['Zero-shot']), fn(groups['Few-shot'])]
        bars = ax.bar(['Zero-shot', 'Few-shot'], vals, color=['#8d99ae', '#2a9d8f'])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    fig.suptitle(f'{TITLE}: zero-shot vs few-shot', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGS, 'zeroshot_vs_fewshot.png'), dpi=150)
    plt.close(fig)


def fig_moral_count_by_dataset(summary):
    datasets = ['allsides', 'basil', 'mpqa']
    mat = np.zeros((len(STRATEGIES), len(datasets)))
    for j, ds in enumerate(datasets):
        sub = summary[(summary['metric'] == 'moral_count') & (summary['dataset'] == ds)]
        s = sub.set_index('strategy')['mean'].reindex(STRATEGIES)
        mat[:, j] = s.values
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels([LABELS[x].replace('\n', ' ') for x in STRATEGIES], fontsize=9)
    for i in range(len(STRATEGIES)):
        for j in range(len(datasets)):
            ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center',
                    color='black' if mat[i, j] < mat.max() * 0.6 else 'white', fontsize=9)
    ax.set_title(f'{TITLE}: moral words preserved by strategy x dataset', fontsize=11)
    fig.colorbar(im, ax=ax, label='moral words (mean)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'moral_count_by_dataset.png'), dpi=150)
    plt.close(fig)


def _sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _bar_color(strat):
    if strat == 'oracle':
        return '#d4a017'
    if strat in FEWSHOT:
        return '#2a9d8f'
    return '#8d99ae'


def fig_paired_contrasts(paired):
    """One PNG per contrast pair, both metrics side by side in each figure."""
    metrics = ['moral_count', 'moral_div']
    directions = {'moral_count': 'higher better', 'moral_div': 'lower better'}

    # collect unique (strat_a, strat_b) pairs
    mc = paired[paired['metric'] == 'moral_count'].reset_index(drop=True)
    pairs = [(r['strat_a'], r['strat_b']) for _, r in mc.iterrows()]

    saved = []
    for strat_a, strat_b in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4.2))
        for ax, metric in zip(axes, metrics):
            row = paired[(paired['metric'] == metric) &
                         (paired['strat_a'] == strat_a) &
                         (paired['strat_b'] == strat_b)]
            if row.empty:
                ax.axis('off')
                continue
            r = row.iloc[0]
            vals = [r['mean_a'], r['mean_b']]
            labels = [strat_a.replace('_', '\n'), strat_b.replace('_', '\n')]
            colors = [_bar_color(strat_a), _bar_color(strat_b)]
            bars = ax.bar([0, 1], vals, color=colors, width=0.55)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels, fontsize=9)
            stars = _sig_stars(r['wilcoxon_p'])
            ax.set_title(f"{metric}  ({directions[metric]})\n"
                         f"n={int(r['n_common'])},  p={r['wilcoxon_p']:.4f}  {stars}",
                         fontsize=9)
            ax.set_ylim(0, max(vals) * 1.28)
            y = max(vals) * 1.10
            ax.plot([0, 1], [y, y], color='black', lw=0.8)
            ax.text(0.5, y, stars, ha='center', va='bottom', fontsize=11)
        fig.suptitle(f'{TITLE}: {strat_a}  vs  {strat_b}\n'
                     '(common articles only — paired Wilcoxon)',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fname = f'paired_{strat_a}_vs_{strat_b}.png'
        fig.savefig(os.path.join(FIGS, fname), dpi=150)
        plt.close(fig)
        saved.append(fname)
    return saved


def main():
    summary = pd.read_csv(os.path.join(DATA, 'strategy_summary.csv'))
    refusal = pd.read_csv(os.path.join(DATA, 'refusal_by_strategy.csv'))
    fig_moral_count(summary)
    fig_moral_div(summary)
    fig_refusal_rate(refusal)
    fig_refusal_subtype(refusal)
    fig_response_breakdown(refusal)
    fig_zeroshot_vs_fewshot(summary, refusal)
    fig_moral_count_by_dataset(summary)

    paired_path = os.path.join(DATA, 'paired_comparison.csv')
    if os.path.isfile(paired_path):
        fig_paired_contrasts(pd.read_csv(paired_path))

    print('Wrote figures to', FIGS)
    for f in sorted(os.listdir(FIGS)):
        print('  -', f)


if __name__ == '__main__':
    main()
