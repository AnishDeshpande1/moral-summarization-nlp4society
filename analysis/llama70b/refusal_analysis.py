"""Classify every Meta-Llama-3-70B-Instruct response and quantify refusals & format failures.

Read-only over results/test_prompts. Writes flat into analysis/gptoss/.

Two kinds of refusal are distinguished for this model:
  - safety refusal       : the model itself declines ("I cannot ...")
  - moderation refusal   : the provider's input filter blocked the prompt
                           (written by prompting.py as [MODERATION_REFUSAL])

Classification per response file:
  ok           : well-formed SUMMARY: ... END OF SUMMARY.
  format_miss  : has a SUMMARY: block but missing the END OF SUMMARY. token
  refusal      : safety refusal or moderation block (no usable summary)
  no_summary   : no SUMMARY: token and not an obvious refusal (malformed)

Usage:
  python analysis/gptoss/refusal_analysis.py
"""
import os
import re
import csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(ROOT, 'results', 'test_prompts')
OUT_DIR = HERE
MODEL_TAG = 'Meta-Llama-3-70B-Instruct'

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot', 'oracle']

REFUSAL_PATTERNS = [
    r"\bI cannot\b", r"\bI can't\b", r"\bI can not\b",
    r"\bI'm not able to\b", r"\bI am not able to\b",
    r"\bI'm unable to\b", r"\bI am unable to\b",
    r"\bI won't\b", r"\bI will not\b",
    r"\bI'm sorry, but\b", r"\bI apologize, but\b",
    r"cannot provide", r"can't provide", r"cannot create", r"can't create",
    r"cannot assist", r"can't assist", r"cannot fulfill", r"can't fulfill",
]
REFUSAL_RE = re.compile('|'.join(REFUSAL_PATTERNS), re.IGNORECASE)


def strategy_of(filename):
    base = filename[len(MODEL_TAG) + 1:] if filename.startswith(MODEL_TAG + '_') else filename
    for strat in sorted(STRATEGIES, key=len, reverse=True):
        if base == f'{strat}_response.txt':
            return strat
    return None


def classify(text):
    if text.startswith('[MODERATION_REFUSAL]'):
        return 'refusal'
    has_summary = 'SUMMARY:' in text
    has_end = 'END OF SUMMARY' in text
    if not has_summary:
        head = text.strip()[:300]
        return 'refusal' if REFUSAL_RE.search(head) else 'no_summary'
    pre = text.split('SUMMARY:', 1)[0]
    if REFUSAL_RE.search(pre[:300]) and not pre.strip().endswith(':'):
        return 'refusal'
    return 'ok' if has_end else 'format_miss'


def refusal_subtype(text):
    """For refusals, mark whether it's moderation (provider) or safety (model)."""
    if text.startswith('[MODERATION_REFUSAL]'):
        return 'moderation'
    return 'safety'


def scan():
    records = []
    for dataset in sorted(os.listdir(RESULTS_DIR)):
        dpath = os.path.join(RESULTS_DIR, dataset)
        if not os.path.isdir(dpath):
            continue
        for article in sorted(os.listdir(dpath)):
            apath = os.path.join(dpath, article)
            if not os.path.isdir(apath):
                continue
            for fname in os.listdir(apath):
                if not (fname.startswith(MODEL_TAG) and fname.endswith('_response.txt')):
                    continue
                strat = strategy_of(fname)
                if strat is None:
                    continue
                with open(os.path.join(apath, fname), 'r', encoding='utf-8') as f:
                    text = f.read()
                cls = classify(text)
                records.append({
                    'dataset': dataset, 'article': article, 'strategy': strat,
                    'classification': cls,
                    'refusal_subtype': refusal_subtype(text) if cls == 'refusal' else '',
                    'char_len': len(text),
                })
    return records


def write_per_response(records):
    path = os.path.join(OUT_DIR, 'response_classifications.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'article', 'strategy',
                                          'classification', 'refusal_subtype', 'char_len'])
        w.writeheader()
        w.writerows(records)
    return path


def aggregate(records):
    by_strat = defaultdict(lambda: defaultdict(int))
    by_strat_dataset = defaultdict(lambda: defaultdict(int))
    for r in records:
        by_strat[r['strategy']][r['classification']] += 1
        by_strat[r['strategy']]['total'] += 1
        if r['classification'] == 'refusal':
            by_strat[r['strategy']]['refusal_' + r['refusal_subtype']] += 1
        key = (r['strategy'], r['dataset'])
        by_strat_dataset[key][r['classification']] += 1
        by_strat_dataset[key]['total'] += 1
    return by_strat, by_strat_dataset


def write_strategy_aggregate(by_strat):
    path = os.path.join(OUT_DIR, 'refusal_by_strategy.csv')
    cats = ['ok', 'refusal', 'format_miss', 'no_summary']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['strategy', 'total'] + cats +
                   ['refusal_safety', 'refusal_moderation', 'refusal_rate', 'unusable_rate'])
        for strat in STRATEGIES:
            d = by_strat.get(strat, {})
            total = d.get('total', 0) or 1
            refusal = d.get('refusal', 0)
            unusable = refusal + d.get('no_summary', 0)
            w.writerow([strat, d.get('total', 0)] +
                       [d.get(c, 0) for c in cats] +
                       [d.get('refusal_safety', 0), d.get('refusal_moderation', 0),
                        round(refusal / total, 3), round(unusable / total, 3)])
    return path


def print_insights(by_strat, by_strat_dataset):
    print('=' * 78)
    print('REFUSAL & FORMAT ANALYSIS  (model: Meta-Llama-3-70B-Instruct, 400 articles/strategy)')
    print('=' * 78)

    print('\n--- per-strategy counts (of 400) ---')
    print(f"{'strategy':<20}{'ok':>5}{'refusal':>9}{'(safety':>9}{'mod)':>6}"
          f"{'fmt_miss':>9}{'no_sum':>8}{'ref%':>8}")
    for strat in STRATEGIES:
        d = by_strat.get(strat, {})
        total = d.get('total', 0) or 1
        print(f"{strat:<20}{d.get('ok',0):>5}{d.get('refusal',0):>9}"
              f"{d.get('refusal_safety',0):>9}{d.get('refusal_moderation',0):>6}"
              f"{d.get('format_miss',0):>9}{d.get('no_summary',0):>8}"
              f"{d.get('refusal',0)/total:>7.1%}")

    print('\n--- zero-shot vs few-shot refusal rate ---')
    def rate(strats):
        ref = sum(by_strat.get(s, {}).get('refusal', 0) for s in strats)
        tot = sum(by_strat.get(s, {}).get('total', 0) for s in strats) or 1
        return ref / tot
    print(f"  zero-shot: {rate(ZEROSHOT):.1%}   few-shot: {rate(FEWSHOT):.1%}")

    print('\n--- moderation vs safety refusals (total) ---')
    mod = sum(by_strat.get(s, {}).get('refusal_moderation', 0) for s in STRATEGIES)
    saf = sum(by_strat.get(s, {}).get('refusal_safety', 0) for s in STRATEGIES)
    print(f"  moderation (provider filter): {mod}")
    print(f"  safety (model declines):      {saf}")

    print('\n--- refusals by dataset ---')
    ds_ref = defaultdict(lambda: [0, 0])
    for (strat, ds), d in by_strat_dataset.items():
        ds_ref[ds][0] += d.get('refusal', 0)
        ds_ref[ds][1] += d.get('total', 0)
    for ds in sorted(ds_ref):
        ref, tot = ds_ref[ds]
        print(f"  {ds:<10} {ref:>3} / {tot:<4} ({ref/(tot or 1):.1%})")


def main():
    records = scan()
    print(f"Scanned {len(records)} response files.")
    p1 = write_per_response(records)
    by_strat, by_strat_dataset = aggregate(records)
    p2 = write_strategy_aggregate(by_strat)
    print_insights(by_strat, by_strat_dataset)
    print(f"\nWrote {os.path.basename(p1)}, {os.path.basename(p2)} -> {OUT_DIR}")


if __name__ == '__main__':
    main()
