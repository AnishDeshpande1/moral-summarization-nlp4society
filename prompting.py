import os
from tqdm import tqdm

from moral_summarization.utils import *
from moral_summarization.args import load_config


# Parse command line arguments and config file
config = load_config(inference=True)

# Select inference backend:
#   "transformers" (default) - load the model in-process with bitsandbytes 4-bit
#                              quantization (DelftBlue / large-VRAM GPUs).
#   "ollama"                 - local GGUF quant via the Ollama server.
#   "openrouter"             - call the OpenRouter hosted API (no GPU needed).
# Set via the `backend:` key in the config file.
backend = config.get('backend', 'transformers')

if backend == 'ollama':
    from moral_summarization.ollama_model import OllamaModelForSequenceCompletion
    llama_model = OllamaModelForSequenceCompletion(config)
elif backend == 'openrouter':
    from moral_summarization.openrouter_model import (
        OpenRouterModelForSequenceCompletion, ModerationRefusal)
    llama_model = OpenRouterModelForSequenceCompletion(config)
else:
    # Imported lazily so the API/Ollama paths don't require torch / peft /
    # bitsandbytes to be installed.
    import torch
    from moral_summarization.model import LlamaModelForSequenceCompletion

    torch.manual_seed(0)
    llama_model = LlamaModelForSequenceCompletion(config)

# Model tag prepended to response filenames so results from different models can
# coexist in one results tree and be selected at eval time (evaluate.py --model).
# Defaults to the base / ollama / openrouter model name; sanitized for filenames.
model_tag = config.get('model_tag') or config.get('base_model_name') \
    or config.get('ollama', {}).get('model') \
    or config.get('openrouter', {}).get('model', 'model')
model_tag = sanitize_model_tag(model_tag)
print(f"Writing responses with model tag: {model_tag}")

# Strategy-first ordering: process one strategy across the WHOLE dataset before
# moving to the next. This lets you finish e.g. the few-shot strategies for all
# 400 articles first, then the others. Override the order/subset via the
# `strategy_order:` config key. Default order requested: few-shot first.
DEFAULT_STRATEGY_ORDER = [
    'cot_fewshot',
    'simple_fewshot',
    'simple',
    'cot',
    'cot_fewshot_mft',
    'simple_fewshot_mft',
    'vanilla',
    'oracle',
]
strategy_order = config.get('strategy_order', DEFAULT_STRATEGY_ORDER)

prompt_dir = config['inference']['prompt_dir']
dataset_folders = sorted(os.listdir(prompt_dir))


def prompt_filename(strategy):
    return f"{strategy}_prompt.txt"


for strategy in strategy_order:
    print(f"\n===== Strategy: {strategy} =====")
    prompt_file = prompt_filename(strategy)

    for dataset_folder in dataset_folders:
        dataset_path = os.path.join(prompt_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        article_folders = sorted(os.listdir(dataset_path))
        for article_folder in tqdm(
                article_folders,
                desc=f"{strategy} | {dataset_folder}"):
            article_path = os.path.join(dataset_path, article_folder)
            if not os.path.isdir(article_path):
                continue

            prompt_path = os.path.join(article_path, prompt_file)
            if not os.path.isfile(prompt_path):
                # This strategy's prompt wasn't generated for this article; skip.
                continue

            # e.g. cot_fewshot_prompt.txt -> {model_tag}_cot_fewshot_response.txt
            response_filename = f"{model_tag}_{prompt_file.replace('prompt', 'response')}"
            response_path = os.path.join(article_path, response_filename)

            # Skip prompts already generated for this model (resumable runs).
            # Delete the response file to force a re-run.
            if os.path.isfile(response_path):
                if config['verbose']:
                    print(f"Skipping existing {article_folder}/{response_filename}")
                continue

            prompt = read_from_file(prompt_path)

            if 'vanilla' in strategy:
                system_content = "You are a news summarizer assistant."
            else:
                system_content = "You are a news summarizer assistant and a moral expert."

            try:
                response, conversation = llama_model.get_response(prompt, system_content)
                content = response[-1]['content']
            except (ModerationRefusal if backend == 'openrouter' else ()) as e:
                # Provider moderation flagged the input - deterministic, so record
                # it as a refusal and move on (writing a file means skip-on-exists
                # won't retry it). Counts as a refusal in the analysis scripts.
                content = f"[MODERATION_REFUSAL] Provider declined input. Reason: {e.reason}"
                if config['verbose']:
                    print(f"Moderation refusal for {article_folder}/{response_filename}: {e.reason}")

            write_to_file(response_path, content)
            if config['verbose']:
                print(f"Generated response for {article_folder}/{response_filename}")

            if config['inference']['testing']:
                # In testing mode, do just one article per dataset per strategy.
                break
