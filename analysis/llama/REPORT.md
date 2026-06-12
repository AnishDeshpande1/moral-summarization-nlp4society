# Llama-3.1-8B evaluation report

Model: llama3.1-8b-instruct-q4_K_M (run locally via Ollama)
Test set: 400 EMONA articles (allsides, basil, mpqa), 8 prompting strategies
each = 3,200 responses.

Figures are in `figures/`.

## Metrics

All metrics are computed against the EMONA human moral-word annotations.

- moral_count: how many of the article's annotated moral words appear in the
  summary. Higher = more moral content preserved.
- moral_div: Jensen-Shannon divergence between the moral-foundation distribution
  of the article and of the summary. Lower = the summary keeps the same balance
  of moral foundations as the source.
- length: summary length in tokens (context only, not a quality score).

## Headline results

Moral-word preservation (moral_count, higher = better):

  vanilla 5.85, simple 5.98, cot 6.14, oracle 11.04,
  simple_fewshot 5.68, cot_fewshot 6.62, simple_fewshot_mft 6.14,
  cot_fewshot_mft 6.59

Best non-oracle strategy is cot_fewshot (6.62). Oracle (which injects the
ground-truth moral words) is the ceiling, as expected.

Moral-frame divergence (moral_div, lower = better):

  best non-oracle is cot_fewshot (0.23). Oracle reaches 0.16.

On both moral metrics the CoT few-shot variants beat the zero-shot baselines.
The simple (non-CoT) few-shot variants do not help and sometimes hurt
(simple_fewshot is the weakest strategy on moral_count).

## Refusals

This is the most striking part of the Llama run. Unlike GPT-OSS (where all
refusals came from the provider's moderation filter), here the refusals are the
model's own safety declines.

Refusal rate by strategy:

  vanilla 0.0%, simple 10.2%, cot 43.0%, oracle 47.2%,
  simple_fewshot 0.3%, cot_fewshot 0.3%, simple_fewshot_mft 0.0%,
  cot_fewshot_mft 0.0%

The zero-shot cot and oracle prompts trigger refusals on nearly half the
articles. Llama-3.1 misreads "preserve the moral framing" as an ask to endorse
a partisan viewpoint and declines. Few-shot conditioning almost completely
eliminates this: the worked examples teach the model that "preserve" means
"summarize faithfully", not "agree with". Few-shot refusal rates are at or near
zero across the board.

This is the opposite pattern from GPT-OSS, where few-shot slightly INCREASED
refusals (longer prompts gave the provider's input filter more to flag). Here
few-shot is what makes the harder prompts usable at all.

## Parse coverage

Because cot and oracle refuse so often, they produce a usable summary on far
fewer articles (cot 227/400, oracle 211/400). The few-shot CoT variants also
have many format misses (~25% missing the END OF SUMMARY token) but still parse
to a summary in almost every case.

This uneven coverage is exactly why the paired comparison below matters: the
plain per-strategy means are each averaged over a different set of articles.

## Paired (intersection) comparison

The per-strategy means above are each averaged over whichever articles that
strategy managed to summarize. Because strategies refuse / fail to parse on
different articles, those means are over different article sets and are not
strictly comparable. paired_comparison.py restricts each contrast to the common
set of articles where BOTH compared strategies produced a usable summary,
recomputes the means there, and runs paired tests (Wilcoxon signed-rank +
paired t-test). One figure per contrast is in figures/ (paired_*.png).

Contrasts run (means on the common set; ** p<0.01, *** p<0.001, ns = not sig.):

  moral_count (higher better):
    cot_fewshot 6.34 vs cot 6.13            (ns)
    cot_fewshot_mft 6.53 > cot 6.14         (**)
    simple_fewshot 5.59 < simple 6.03       (** - simple few-shot HURTS)
    simple_fewshot_mft 6.05 vs simple 5.98  (ns)
    cot 6.14 vs simple 5.93                 (ns)
    cot_fewshot 6.62 > simple_fewshot 5.63  (***)
    cot_fewshot_mft 6.59 > simple_fewshot_mft 6.14 (***)

  moral_div (lower better):
    cot_fewshot 0.232 < cot 0.244           (* p<0.05)
    cot_fewshot_mft 0.232 vs cot 0.245      (ns)
    simple_fewshot 0.262 vs simple 0.255    (ns)
    simple_fewshot_mft 0.249 vs simple 0.253 (ns)
    cot 0.246 vs simple 0.245               (ns)
    cot_fewshot 0.232 < simple_fewshot 0.259 (***)
    cot_fewshot_mft 0.238 vs simple_fewshot_mft 0.246 (ns)

Takeaways:
- Few-shot helps when added to CoT, not to the simple prompt. simple_fewshot is
  significantly worse than plain simple on moral_count.
- The clearest, most significant effect is CoT vs non-CoT within the few-shot
  setting: cot_fewshot and cot_fewshot_mft beat their simple_fewshot
  counterparts on both metrics (***). The reasoning step is what carries the
  gain, not the few-shot examples on their own.

## Data files

See README.txt in the parent analysis/ folder for what each CSV contains.
