"""Summary-quality evaluation with SummaC (reference-free factual consistency).

SummaC scores how well a summary is *entailed* by its source article using an
NLI model (0-1, higher = more faithful). This is one of the reference-free
metrics built into the original Liscio et al. evaluator (eval.py: extra_metrics
= 'summaC', 'BLANC', 'QaFactEval'); here it is pulled out into a standalone
script so it can be run per-model without re-running the whole evaluation.

Read-only over results/test_prompts. For each model it scores every parseable
summary against its source article and writes a tidy pickle:

    columns: dataset, article, strategy, summac
    one row per (dataset, article, strategy)

The output pickle is a long-form DataFrame, mirroring per_article_long.csv so it
slots straight into the existing analysis. It is written into the model's
analysis folder (analysis/llama/ or analysis/gptoss/).

GPU: SummaC runs an NLI model. If a CUDA GPU is visible it is used automatically
(much faster); otherwise it falls back to CPU. Override with --device.

Usage:
    # local Llama run
    python analysis/quality_eval.py --model llama3.1-8b-instruct-q4_K_M --out llama

    # GPT-OSS run
    python analysis/quality_eval.py --model GPT-OSS-120B --out gptoss

    # force CPU / GPU
    python analysis/quality_eval.py --model GPT-OSS-120B --out gptoss --device cpu

Install:
    pip install summac
"""
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from moral_summarization.data_utils import load_article, get_summary_from_response

RESULTS_DIR = os.path.join(ROOT, 'results', 'test_prompts')

STRATEGIES = [
    'vanilla', 'simple', 'cot', 'oracle',
    'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft',
]


def pick_device(requested):
    """Return 'cuda' if available (and not overridden to cpu), else 'cpu'."""
    if requested == 'cpu':
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    if requested == 'cuda':
        print('WARNING: --device cuda requested but no CUDA GPU is visible; '
              'falling back to CPU.')
    return 'cpu'


def strategy_of(filename, model_tag):
    """Map '<tag>_<strategy>_response.txt' -> strategy (exact, longest-first)."""
    base = filename[len(model_tag) + 1:] if filename.startswith(model_tag + '_') else filename
    for strat in sorted(STRATEGIES, key=len, reverse=True):
        if base == f'{strat}_response.txt':
            return strat
    return None


def collect_pairs(model_tag):
    """Walk results/, return (scorable, no_summary).

    scorable   : list of (dataset, article, strategy, article_text, summary) for
                 responses that parse to a non-empty summary -> sent to SummaC.
    no_summary : list of (dataset, article, strategy) for refusals / malformed
                 responses with no extractable summary -> scored as -1 (a
                 sentinel meaning "no summary to evaluate"), NOT sent to SummaC.
    """
    scorable, no_summary = [], []
    article_cache = {}
    for dataset in sorted(os.listdir(RESULTS_DIR)):
        dpath = os.path.join(RESULTS_DIR, dataset)
        if not os.path.isdir(dpath):
            continue
        for article in sorted(os.listdir(dpath)):
            apath = os.path.join(dpath, article)
            if not os.path.isdir(apath):
                continue
            for fname in os.listdir(apath):
                if not (fname.startswith(model_tag) and fname.endswith('_response.txt')):
                    continue
                strat = strategy_of(fname, model_tag)
                if strat is None:
                    continue
                summary = get_summary_from_response(os.path.join(apath, fname))
                if not summary:
                    no_summary.append((dataset, article, strat))
                    continue
                key = (article, dataset)
                if key not in article_cache:
                    article_cache[key] = load_article(article, dataset)
                scorable.append((dataset, article, strat, article_cache[key], summary))
    return scorable, no_summary


SUMMAC_CONV_URL = ('https://github.com/tingofurro/summac/raw/master/'
                   'summac_conv_vitc_sent_perc_e.bin')


def ensure_conv_weights():
    """Return a local path to the SummaC conv head, downloading it if needed.

    SummaC's start_file='default' shells out to `wget`, which doesn't exist on
    Windows. We fetch the (tiny ~2KB) trained MLP head ourselves with urllib and
    cache it under ~/.cache/summac, then hand the path to SummaCConv.
    """
    import urllib.request
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'summac')
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, 'summac_conv_vitc_sent_perc_e.bin')
    if not (os.path.isfile(dest) and os.path.getsize(dest) > 1000):
        print(f"Downloading SummaC conv head -> {dest}")
        urllib.request.urlretrieve(SUMMAC_CONV_URL, dest)
    return dest


def ensure_nltk_punkt():
    """SummaC uses nltk.sent_tokenize for sentence granularity. Make sure the
    punkt tokenizer data is present (name changed to 'punkt_tab' in newer nltk).
    """
    import nltk
    for resource in ('punkt_tab', 'punkt'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
            return
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                nltk.data.find(f'tokenizers/{resource}')
                return
            except LookupError:
                continue


def build_summac(device):
    """Construct the SummaCConv scorer (same config the original evaluator uses)."""
    from summac.model_summac import SummaCConv
    ensure_nltk_punkt()
    start_file = ensure_conv_weights()
    return SummaCConv(
        models=["vitc"], bins='percentile', granularity="sentence",
        nli_labels="e", device=device, start_file=start_file, agg="mean",
    )


def score_pairs(scorer, pairs, batch_size):
    """Score (article, summary) pairs with SummaC, batched. Returns list of floats."""
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc='SummaC'):
        chunk = pairs[i:i + batch_size]
        articles = [p[3] for p in chunk]
        summaries = [p[4] for p in chunk]
        out = scorer.score(articles, summaries)
        scores.extend(out['scores'])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help="Model tag as it appears in response filenames, "
                         "e.g. 'GPT-OSS-120B' or 'llama3.1-8b-instruct-q4_K_M'.")
    ap.add_argument('--out', required=True,
                    help="Output subfolder under analysis/, e.g. 'llama' or 'gptoss'.")
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                    help="Compute device. 'auto' uses GPU if available (default).")
    ap.add_argument('--batch-size', type=int, default=16,
                    help="SummaC scoring batch size (default 16).")
    args = ap.parse_args()

    device = pick_device(args.device)
    out_dir = os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model tag : {args.model}")
    print(f"Device    : {device}")
    print(f"Output    : {out_dir}")

    print("\nCollecting (article, summary) pairs ...")
    scorable, no_summary = collect_pairs(args.model)
    if not scorable and not no_summary:
        raise SystemExit(
            f"No responses found for model tag '{args.model}'. "
            f"Check the tag matches the response filenames in {RESULTS_DIR}.")
    print(f"  {len(scorable)} usable summaries to score.")
    print(f"  {len(no_summary)} responses with no summary (scored -1).")

    rows = []
    if scorable:
        print("\nLoading SummaC (downloads the vitc NLI model on first run) ...")
        scorer = build_summac(device)
        scores = score_pairs(scorer, scorable, args.batch_size)
        for p, s in zip(scorable, scores):
            rows.append({'dataset': p[0], 'article': p[1], 'strategy': p[2],
                         'summac': round(float(s), 4)})

    # Refusals / malformed: no summary to evaluate -> sentinel -1
    for d, a, strat in no_summary:
        rows.append({'dataset': d, 'article': a, 'strategy': strat, 'summac': -1.0})

    df = pd.DataFrame(rows, columns=['dataset', 'article', 'strategy', 'summac'])
    df = df.sort_values(['dataset', 'article', 'strategy']).reset_index(drop=True)

    pkl_path = os.path.join(out_dir, 'summary_quality.pickle')
    csv_path = os.path.join(out_dir, 'summary_quality.csv')
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {len(df)} rows ->")
    print(f"  {pkl_path}")
    print(f"  {csv_path}")

    print("\n--- mean SummaC by strategy (higher = more faithful) ---")
    print("    (-1 rows = no summary; excluded from the mean, counted separately)")
    scored = df[df['summac'] >= 0]
    mean_scored = scored.groupby('strategy')['summac'].mean()
    n_scored = scored.groupby('strategy')['summac'].count()
    n_missing = df[df['summac'] < 0].groupby('strategy')['summac'].count()
    out = pd.DataFrame({
        'mean_summac': mean_scored.round(4),
        'n_scored': n_scored,
        'n_no_summary': n_missing,
    }).reindex(STRATEGIES).fillna(0)
    out['n_scored'] = out['n_scored'].astype(int)
    out['n_no_summary'] = out['n_no_summary'].astype(int)
    print(out.to_string())


if __name__ == '__main__':
    main()
