import os
from tqdm import tqdm

from moral_summarization.utils import *
from moral_summarization.args import load_config


# Parse command line arguments and config file
config = load_config(inference=True)

# Select inference backend. "ollama" runs a local GGUF quant via the Ollama
# server (good for low-VRAM machines and trivial to install). "transformers"
# (default) loads the model in-process with bitsandbytes 4-bit quantization
# (used on DelftBlue / large-VRAM GPUs). Set via the `backend:` key in the
# config file.
backend = config.get('backend', 'transformers')

if backend == 'ollama':
    from moral_summarization.ollama_model import OllamaModelForSequenceCompletion
    llama_model = OllamaModelForSequenceCompletion(config)
else:
    # Imported lazily so the local Ollama path doesn't require torch / peft /
    # bitsandbytes to be installed.
    import torch
    from moral_summarization.model import LlamaModelForSequenceCompletion

    torch.manual_seed(0)
    llama_model = LlamaModelForSequenceCompletion(config)

# Model tag prepended to response filenames so results from different models can
# coexist in one results tree and be selected at eval time (evaluate.py --model).
# Defaults to the base/ollama model name; sanitized for use in filenames.
model_tag = config.get('model_tag') or config.get('base_model_name') \
    or config.get('ollama', {}).get('model', 'model')
model_tag = sanitize_model_tag(model_tag)
print(f"Writing responses with model tag: {model_tag}")

for dataset_folder in os.listdir(config['inference']['prompt_dir']):
    dataset_path = os.path.join(config['inference']['prompt_dir'], dataset_folder)
    for article_folder in tqdm(os.listdir(dataset_path), desc=f"Generating responses for {dataset_folder}"):
        article_path = os.path.join(dataset_path, article_folder)
        for file_path in os.listdir(article_path):
            prompt_path = os.path.join(article_path, file_path)
            if 'prompt' in file_path and os.path.isfile(prompt_path):
                prompt = read_from_file(prompt_path)

                if 'vanilla' in file_path:
                    system_content = "You are a news summarizer assistant."
                else:
                    system_content = "You are a news summarizer assistant and a moral expert."

                # e.g. simple_prompt.txt -> {model_tag}_simple_response.txt
                response_filename = f"{model_tag}_{file_path.replace('prompt', 'response')}"
                response_path = os.path.join(article_path, response_filename)

                # Skip prompts already generated for this model. Makes runs
                # resumable: interrupted local runs and cluster jobs that hit the
                # time limit pick up where they left off instead of redoing work
                # (and overwriting). Delete the response file to force a re-run.
                if os.path.isfile(response_path):
                    if config['verbose']:
                        print(f"Skipping existing {article_folder}/{response_filename}")
                    continue

                response, conversation = llama_model.get_response(prompt, system_content)

                write_to_file(response_path, response[-1]['content'])
                if config['verbose']:
                    print(f"Generated response for {article_folder}/{file_path}")

        if config['inference']['testing']:
            break
