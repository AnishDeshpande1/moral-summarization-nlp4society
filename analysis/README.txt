analysis/
=========

Read-only analysis of the summarization evaluation. Nothing here modifies
results/test_prompts/ or the EMONA data. The scripts only read those and write
derived CSVs, a report, and figures.

There is one set of scripts for the local Llama-3.1-8B run (in analysis/ itself)
and a parallel set for the GPT-OSS-120B run (in analysis/gptoss/). They do the
same thing for different models.


Scripts
-------

analyze_core_metrics.py
  Loads the evaluator pickle (core_metrics.pickle) and builds per-strategy
  comparison tables for the three metrics. Prints insights and writes CSVs.

refusal_analysis.py
  Reads every response file and labels each one: ok, refusal, format_miss, or
  no_summary. Counts refusal rates per strategy and per dataset. The GPT-OSS
  version also splits refusals into "safety" (the model declines) vs
  "moderation" (the provider's input filter blocked the prompt).

make_plots.py
  Renders the figures from the CSVs into the figures/ folder. Needs matplotlib.

paired_comparison.py  (gptoss/ only so far)
  The plain per-strategy means are each averaged over whichever articles that
  strategy managed to summarize. Since strategies refuse on different articles,
  those means are over different article sets and are not directly comparable.
  This script restricts to the common set of articles where every compared
  strategy produced a usable summary, recomputes the means there, and runs
  paired tests (Wilcoxon signed-rank and paired t-test) on the key contrasts.
  This is the fair, apples-to-apples comparison. Needs scipy.


How to run
----------

From the repo root, after run_evaluation.py has produced dict_of_dfs.pickle:

  Llama-3.1-8B:
    cp dict_of_dfs.pickle analysis/data/core_metrics.pickle
    python analysis/analyze_core_metrics.py
    python analysis/refusal_analysis.py
    python analysis/make_plots.py

  GPT-OSS-120B:
    python run_evaluation.py --results-dir results/test_prompts --model GPT-OSS-120B
    cp dict_of_dfs.pickle analysis/gptoss/core_metrics.pickle
    python analysis/gptoss/analyze_core_metrics.py
    python analysis/gptoss/refusal_analysis.py
    python analysis/gptoss/make_plots.py


The three metrics
-----------------

All are computed against the EMONA human moral-word annotations.

  moral_count  How many of the article's annotated moral words appear in the
               summary. Higher means more moral content preserved.

  moral_div    Jensen-Shannon divergence between the moral-foundation
               distribution of the article and of the summary. Lower means the
               summary keeps the same balance of moral foundations as the source.

  length       Summary length in tokens. Context only, not a quality score.


Data files written
------------------

  core_metrics.pickle          Raw evaluator output (copy of dict_of_dfs.pickle).
                               Three dataframes: moral_count, moral_div, length,
                               indexed by model / dataset / article, with one
                               column per strategy plus an "original" column.

  strategy_summary.csv         Mean value per metric, strategy, and dataset
                               (plus an "all" row that averages the datasets).
                               This is the main table behind the bar charts.

  per_article_long.csv         Tidy long form: one row per article, strategy,
                               and metric, with the raw value. Use this for any
                               custom stats or significance tests.

  parse_coverage.csv           Per strategy: how many of the 400 articles
                               produced a usable, parseable summary, and the
                               parse-failure rate.

  response_classifications.csv One row per response file with its label
                               (ok / refusal / format_miss / no_summary). The
                               GPT-OSS version adds a refusal_subtype column
                               (safety vs moderation).

  refusal_by_strategy.csv      Per strategy: counts of each label and the
                               refusal rate. The GPT-OSS version also splits
                               safety vs moderation refusals.

  paired_comparison.csv        Key strategy-vs-strategy contrasts computed on the
                               common (intersection) article set, with paired
                               Wilcoxon and t-test p-values. This is the fair
                               comparison (same articles for both strategies).


Figures written (figures/)
--------------------------

  moral_count_by_strategy.png      Bar chart of moral words preserved per
                                   strategy, with the oracle ceiling shown.
                                   Higher is better.

  moral_div_by_strategy.png        Bar chart of moral-frame divergence per
                                   strategy. Lower is better.

  refusal_rate_by_strategy.png     Refusal rate per strategy.

  response_breakdown_stacked.png   Stacked bar of the four response outcomes
                                   (ok / format_miss / refusal / no_summary) per
                                   strategy. Shows the whole picture at a glance.

  zeroshot_vs_fewshot.png          Three side-by-side panels comparing the
                                   zero-shot baselines against the few-shot
                                   strategies on moral words, frame divergence,
                                   and refusal rate.

  moral_count_by_dataset.png       Heatmap of moral words preserved, strategy by
                                   dataset (allsides / basil / mpqa).

  refusal_subtype_by_strategy.png  (GPT-OSS only) Stacked bar splitting refusals
                                   into safety (model declines) vs moderation
                                   (provider input filter). For GPT-OSS every
                                   refusal is a moderation block.

  paired_contrasts.png             (GPT-OSS only) The fair comparison: few-shot
                                   strategy vs its baseline, means computed on
                                   the common articles only, with the common-set
                                   size n and significance stars from the paired
                                   tests. Built from paired_comparison.csv.

Color coding in the bar charts: grey = zero-shot baselines, gold = oracle
ceiling, teal = few-shot strategies.


Reports
-------

  reports/REPORT.md     Write-up of the local Llama-3.1-8B run.
  gptoss/REPORT.md      Write-up of the GPT-OSS-120B run.


Per-model folders
-----------------

Each model has its own folder with the same parallel set of scripts and outputs:

  llama/        Llama-3.1-8B-Instruct (q4_K_M, local Ollama). Full 400 articles.
  gptoss/       GPT-OSS-120B (API). Full 400 articles.
  llama70b/     Meta-Llama-3-70B-Instruct. PARTIAL run: allsides 180/180,
                basil 34/150, mpqa absent (~214 usable articles/strategy).
  deepseek/     DeepSeek-R1-Distill-Qwen-32B. PARTIAL run: allsides only, and
                only 52/180 articles. CoT variants drop to 25-45 usable articles
                because R1's reasoning traces often never emit the SUMMARY: token.

The llama70b/ and deepseek/ means are therefore over far fewer articles than the
two full runs, and are not directly comparable to them in absolute terms; read
them per-strategy WITHIN a model (the paired contrasts are the fair comparison).


Adding other models
-------------------

Each script has a MODEL_TAG constant near the top (and make_plots.py a TITLE)
and writes to a fixed folder. To analyze another model:

  1. cp -r analysis/gptoss analysis/<newmodel>
  2. In the 4 scripts, replace the 'GPT-OSS-120B' tag with the new model's
     response-file tag (refusal_analysis.py MODEL_TAG must match exactly; the
     others only use it in printed labels). Set make_plots.py TITLE to a short
     display name.
  3. python run_evaluation.py --model <tag>   (or a fixed flag: --llama for
     Meta-Llama-3-70B-Instruct, --deepseek for DeepSeek-R1-Distill-Qwen-32B,
     --command-r for c4ai-command-r-plus-4bit). This writes dict_of_dfs.pickle.
  4. cp dict_of_dfs.pickle analysis/<newmodel>/core_metrics.pickle
  5. Run the four scripts in order: analyze_core_metrics.py, refusal_analysis.py,
     paired_comparison.py, make_plots.py.

For a partial run (not all datasets/articles present), nothing special is
needed -- the scripts read whatever response files exist and the parse-coverage
table reports how many articles each strategy actually covered.
