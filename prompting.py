import os
import torch
from tqdm import tqdm

from moral_summarization.utils import *
from moral_summarization.args import load_config
from moral_summarization.model import LlamaModelForSequenceCompletion


# Parse command line arguments and config file
config = load_config(inference=True)

torch.manual_seed(0)

llama_model = LlamaModelForSequenceCompletion(config)

for dataset_folder in os.listdir(config['inference']['prompt_dir']):
    dataset_path = os.path.join(config['inference']['prompt_dir'], dataset_folder)
    if not os.path.isdir(dataset_path):
        continue
    for article_folder in tqdm(os.listdir(dataset_path), desc=f"Generating responses for {dataset_folder}"):
        article_path = os.path.join(dataset_path, article_folder)
        if not os.path.isdir(article_path):
            continue
        for file_path in os.listdir(article_path):
            prompt_path = os.path.join(article_path, file_path)
            if 'prompt' in file_path and os.path.isfile(prompt_path):
                prompt = read_from_file(prompt_path)

                if 'vanilla' in file_path:
                    system_content = "You are a news summarizer assistant. Do not include any introductory text, polite chatter, conversational filler, or final remarks. Output only the requested format directly."
                else:
                    system_content = "You are a news summarizer assistant and a moral expert. Do not include any introductory text, polite chatter, conversational filler, or final remarks. Output only the requested format directly."

                if 'cot' in file_path and config['inference'].get('use_chaining', False):
                    # --- CALL 1: EXTRACTION ---
                    step1_instruction = (
                        "(1) First, you identify all the single words that are morally framed. "
                        "Identify this step as \"STEP 1:\" and report each word in a new line starting with *\n"
                        "Do not write the summary or output any other text. Only output the STEP 1 list of words directly."
                    )
                    idx = prompt.find("(1) First, you identify")
                    step1_prompt = prompt[:idx] + step1_instruction if idx != -1 else prompt + "\nOnly perform STEP 1."

                    extracted_words, _ = llama_model.get_response(step1_prompt, system_content)
                    
                    if "STEP 1:" not in extracted_words:
                        extracted_words = "STEP 1:\n" + extracted_words.strip()

                    # --- CALL 2: SUMMARIZATION ---
                    step2_instruction = (
                        f"\nHere are the extracted morally framed words for this article:\n{extracted_words}\n\n"
                        "Now, perform Step 2: write the summary of the news article (maximum 200 words) preserving the moral framing. "
                        "Naturally weave in the extracted moral words. "
                        "The summary has to be returned after a \"SUMMARY:\" token and ending with a \"END OF SUMMARY.\" token."
                    )
                    step2_prompt = prompt[:idx] + step2_instruction if idx != -1 else prompt + step2_instruction

                    summary_text, _ = llama_model.get_response(step2_prompt, system_content)

                    if "SUMMARY:" not in summary_text:
                        summary_text = "SUMMARY:\n" + summary_text.strip()

                    response_text = f"{extracted_words.strip()}\n\n{summary_text.strip()}"
                else:
                    response, conversation = llama_model.get_response(prompt, system_content)
                    response_text = response if isinstance(response, str) else response[-1]['content']
                
                if config['inference']['use_ollama']:
                    model_name = config['inference']['ollama_model']
                else:
                    model_name = llama_model.base_model_name.split('/')[-1]

                response_path = os.path.join(article_path, file_path.replace('_prompt.txt', f'_response_{model_name}.txt'))
                
                # Remove any conversational prefixes before the first actual token (STEP 1: or SUMMARY:)
                clean_response = response_text
                first_token_idx = None
                for token in ["STEP 1:", "STEP 1**", "**STEP 1:**", "SUMMARY:", "SUMMARY**", "**SUMMARY:**", "Summary:", "**Summary:**"]:
                    idx = clean_response.find(token)
                    if idx != -1:
                        if first_token_idx is None or idx < first_token_idx:
                            first_token_idx = idx
                if first_token_idx is not None:
                    clean_response = clean_response[first_token_idx:].strip()

                write_to_file(response_path, clean_response)
                if config['verbose']:
                    print(f"Generated response for {article_folder}/{file_path}")

        if config['inference']['testing']:
            break
