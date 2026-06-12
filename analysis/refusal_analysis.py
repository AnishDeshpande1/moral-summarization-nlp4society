"""Classify every generated response and quantify refusals & format failures.

Read-only over results/test_prompts. Writes per-response classifications and
per-strategy / per-dataset aggregates to analysis/data/.

Why this matters: the prompts are used verbatim from Liscio et al. (plus our
few-shot additions). Safety-tuned Llama-3.1 sometimes *refuses* to summarize
sensitive articles when asked to "preserve the moral framing" - it misreads the
instruction as an ask to endorse the framing. This is a faithful, reportable
outcome, and a robustness signal: which prompting strategies trigger refusals,
and does our few-shot conditioning reduce them?

Classification (per response file):
  refusal      : model declined (e.g. "I cannot provide...") - no summary at all
  no_summary   : no SUMMARY: token and not an obvious refusal (malformed output)
  format_miss  : has a SUMMARY: block but missing the END OF SUMMARY. token
  ok           : well-formed SUMMARY: ... END OF SUMMARY.

Usage:
  python analysis/refusal_analysis.py
"""
import os
import re
import csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results', 'test_prompts')
OUT_DIR = os.path.join(ROOT, 'analysis', 'llama')
MODEL_TAG = 'llama3.1-8b-instruct-q4_K_M'

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]
FEWSHOT = ['simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
ZEROSHOT = ['vanilla', 'simple', 'cot', 'oracle']

# Phrases that mark a safety / capability refusal. Kept conservative: these are
# the standard Llama refusal openers, matched near the start of the response.
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
    """Map '<tag>_<strategy>_response.txt' -> strategy, exact-match aware.

    Longest strategy names first so 'cot_fewshot_mft' wins over 'cot'.
    """
    base = filename[len(MODEL_TAG) + 1:] if filename.startswith(MODEL_TAG + '_') else filename
    for strat in sorted(STRATEGIES, key=len, reverse=True):
        if base == f'{strat}_response.txt':
            return strat
    return None


def classify(text):
    # Provider-side moderation refusal written by prompting.py.
    if text.startswith('[MODERATION_REFUSAL]'):
        return 'refusal'

    has_summary = 'SUMMARY:' in text
    has_end = 'END OF SUMMARY' in text

    if not has_summary:
        # No summary block. Refusal if it opens with a refusal phrase, else malformed.
        head = text.strip()[:300]
        if REFUSAL_RE.search(head):
            return 'refusal'
        return 'no_summary'
    # Has a SUMMARY block. Still flag if it also contains a refusal before SUMMARY.
    pre = text.split('SUMMARY:', 1)[0]
    if REFUSAL_RE.search(pre[:300]) and not pre.strip().endswith(':'):
        # refusal text appears before a (possibly templated) SUMMARY token
        return 'refusal'
    if not has_end:
        return 'format_miss'
    return 'ok'


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
                    continue  # e.g. 'class' responses, not in our 8
                with open(os.path.join(apath, fname), 'r', encoding='utf-8') as f:
                    text = f.read()
                records.append({
                    'dataset': dataset,
                    'article': article,
                    'strategy': strat,
                    'classification': classify(text),
                    'char_len': len(text),
                })
    return records


def write_per_response(records):
    path = os.path.join(OUT_DIR, 'response_classifications.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'article', 'strategy',
                                          'classification', 'char_len'])
        w.writeheader()
        w.writerows(records)
    return path


def aggregate(records):
    # counts[strategy][classification] and counts[strategy]['total']
    by_strat = defaultdict(lambda: defaultdict(int))
    by_strat_dataset = defaultdict(lambda: defaultdict(int))
    for r in records:
        by_strat[r['strategy']][r['classification']] += 1
        by_strat[r['strategy']]['total'] += 1
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
                   ['refusal_rate', 'unusable_rate'])
        for strat in STRATEGIES:
            d = by_strat.get(strat, {})
            total = d.get('total', 0) or 1
            refusal = d.get('refusal', 0)
            unusable = refusal + d.get('no_summary', 0)  # no extractable summary
            w.writerow([strat, d.get('total', 0)] +
                       [d.get(c, 0) for c in cats] +
                       [round(refusal / total, 3), round(unusable / total, 3)])
    return path


def print_insights(by_strat, by_strat_dataset):
    print('=' * 78)
    print('REFUSAL & FORMAT ANALYSIS  (model: %s, 400 articles/strategy)' % MODEL_TAG)
    print('=' * 78)

    print('\n--- per-strategy classification counts (of 400) ---')
    hdr = f"{'strategy':<20}{'ok':>5}{'refusal':>9}{'fmt_miss':>9}{'no_summ':>9}{'refusal%':>10}"
    print(hdr)
    for strat in STRATEGIES:
        d = by_strat.get(strat, {})
        total = d.get('total', 0) or 1
        print(f"{strat:<20}{d.get('ok',0):>5}{d.get('refusal',0):>9}"
              f"{d.get('format_miss',0):>9}{d.get('no_summary',0):>9}"
              f"{d.get('refusal',0)/total:>9.1%}")

    print('\n--- zero-shot vs few-shot refusal rate ---')
    def rate(strats):
        ref = sum(by_strat.get(s, {}).get('refusal', 0) for s in strats)
        tot = sum(by_strat.get(s, {}).get('total', 0) for s in strats) or 1
        return ref / tot
    print(f"  zero-shot ({', '.join(ZEROSHOT)}): {rate(ZEROSHOT):.1%}")
    print(f"  few-shot  ({', '.join(FEWSHOT)}): {rate(FEWSHOT):.1%}")
    if rate(FEWSHOT) < rate(ZEROSHOT):
        print("  > Few-shot conditioning REDUCES refusals "
              "(worked examples teach 'preserve' != 'endorse').")
    else:
        print("  > Few-shot did not reduce refusals.")

    print('\n--- refusals by dataset (which topics trigger declines) ---')
    ds_ref = defaultdict(lambda: [0, 0])  # dataset -> [refusals, total]
    for (strat, ds), d in by_strat_dataset.items():
        ds_ref[ds][0] += d.get('refusal', 0)
        ds_ref[ds][1] += d.get('total', 0)
    for ds in sorted(ds_ref):
        ref, tot = ds_ref[ds]
        print(f"  {ds:<10} {ref:>3} / {tot:<4} ({ref/(tot or 1):.1%})")

    print('\n--- strategies most prone to refusal (ranked) ---')
    ranked = sorted(STRATEGIES,
                    key=lambda s: by_strat.get(s, {}).get('refusal', 0),
                    reverse=True)
    for s in ranked[:4]:
        print(f"  {s:<20} {by_strat.get(s, {}).get('refusal', 0)} refusals")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = scan()
    print(f"Scanned {len(records)} response files.")
    p1 = write_per_response(records)
    by_strat, by_strat_dataset = aggregate(records)
    p2 = write_strategy_aggregate(by_strat)
    print_insights(by_strat, by_strat_dataset)
    print(f"\nWrote: {os.path.basename(p1)}, {os.path.basename(p2)} -> {OUT_DIR}")


if __name__ == '__main__':
    main()
