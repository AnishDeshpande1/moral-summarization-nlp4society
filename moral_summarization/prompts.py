import os
import pandas as pd
from ast import literal_eval

from .utils import *
from .data_utils import *

vanilla_intro = 'You have to summarize the following article.'
moral_intro = 'You have to summarize the following text preserving the moral framing that the author gave to it.'
here_is_the_news_article = 'Here is the news article:\n\n'
cot_instructions = '(1) First, you identify all the single words that are morally framed. Identify this step as "STEP 1:" and report each word in a new line starting with *\n(2) Finally, you write a summary of the news article. '
preserve_moral_words = 'Please preserve as many morally framed words as possible in your summary. '
closing = 'The summary has to be returned after a "SUMMARY:" token and ending with a "END OF SUMMARY." token. The summary should be no longer than 200 words.'

# Lazy loading of ExemplarSelector to avoid loading it on every import
exemplar_selector = None

def get_exemplar_selector():
    global exemplar_selector
    if exemplar_selector is None:
        from moral_summarization.select_exemplars import ExemplarSelector
        exemplar_selector = ExemplarSelector()
    return exemplar_selector


def load_exemplar_response(article_name, dataset, prompt_type, model='Meta-Llama-3-70B-Instruct', seed='345'):
    """
    Load the pre-computed response file for the given exemplar article.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    response_folder = os.path.join(root_dir, 'results', 'summaries', dataset, article_name)
    
    # Try to find the response file matching exact prompt_type, model, and seed
    file_name = f"{prompt_type}_response_{model}_{seed}.txt"
    file_path = os.path.join(response_folder, file_name)
    
    if os.path.exists(file_path):
        return read_from_file(file_path)
    
    # Fallback to any file matching prompt_type and model
    if os.path.exists(response_folder):
        files = os.listdir(response_folder)
        matching = [f for f in files if prompt_type in f and model in f and 'response' in f]
        if matching:
            return read_from_file(os.path.join(response_folder, matching[0]))
        # Fallback to any matching prompt_type response file
        matching_any = [f for f in files if prompt_type in f and 'response' in f]
        if matching_any:
            return read_from_file(os.path.join(response_folder, matching_any[0]))
            
    print(f"Warning: Could not find pre-computed exemplar response for {article_name} in {response_folder}")
    return None


def load_predictions_df(predictions_path):
    literal_eval_columns = ['predicted_words', 'labeled_words']
    converters = {column: literal_eval for column in literal_eval_columns}
    return pd.read_csv(predictions_path, converters=converters)


def get_clean_article_text(article_path):
    article_text = read_from_file(article_path)
    return article_text.replace('\n', ' ').replace('  ', ' ')


def make_article_prompt(article_text):
    return here_is_the_news_article + article_text


def load_moral_annotations(dataset, article_file):
    annotations_path = os.path.join(EMONA_dataset_path, annotation_folders[dataset], f'{article_file}.json')
    annotations = load_json(annotations_path)
    moral_annotations = get_moral_annotations(annotations)
    return [remove_punctuation(annotation['token']) for annotation in moral_annotations]


def load_predicted_moral_words(predictions_df, article_file):
    predicted_words = []
    # loop through the sentences of this article and get the predicted words
    for _, row in predictions_df[predictions_df['article'] == article_file].iterrows():
        predicted_words.extend(row['predicted_words'])

    return predicted_words


def make_moral_words_list(moral_words, deduplicate=True):
    intro = 'The author used the following morally framed words in the article:\n'

    if deduplicate:
        moral_words = list(dict.fromkeys(moral_words))

    # print the words in a list with a bullet point
    moral_list = '\n'.join([f'* {word}' for word in moral_words])

    return intro + moral_list


def dump_prompt(prompt, destination_folder, prompt_type):
    destination_file = os.path.join(destination_folder, f'{prompt_type}_prompt.txt')
    write_to_file(destination_file, prompt)


def dump_vanilla_prompt(article_prompt, destination_folder):
    prompt = vanilla_intro + '\n\n' + article_prompt + '\n\n' + closing
    dump_prompt(prompt, destination_folder, 'vanilla')


def dump_simple_prompt(article_prompt, destination_folder):
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + closing
    dump_prompt(prompt, destination_folder, 'simple')


def dump_cot_prompt(article_prompt, destination_folder):
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + cot_instructions \
          + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'cot')


def dump_oracle_prompt(article_prompt, destination_folder, dataset, article_file, deduplicate=True):
    moral_annotations = load_moral_annotations(dataset, article_file)
    moral_list = make_moral_words_list(moral_annotations, deduplicate)
    prompt = moral_intro + '\n\n' + article_prompt + '\n\n' + moral_list + '\n\n' \
         + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'oracle')


def dump_class_prompt(article_prompt, destination_folder, predicions_df, article_file, deduplicate=True):
    # If the article is not in the test set, do not generate a prompt
    if article_file not in predicions_df['article'].to_list():
        return

    moral_predictions = load_predicted_moral_words(predicions_df, article_file)
    moral_list = make_moral_words_list(moral_predictions, deduplicate)
    prompt = moral_intro + article_prompt + '\n\n' + moral_list + '\n\n' \
         + preserve_moral_words + closing
    dump_prompt(prompt, destination_folder, 'class')


def dump_simple_fewshot_prompt(article_text, target_article_name, dataset, destination_folder, num_shots=3):
    selector = get_exemplar_selector()
    triplet = selector.select_exemplars(article_text, target_article_name)
    
    prompt = moral_intro + '\n\n'
    prompt += "Please study the following examples of news articles and their summaries preserving the moral framing:\n\n"
    
    ideology_names = { -1: "Left", 0: "Center", 1: "Right" }
    
    for i, exemplar in enumerate(triplet, 1):
        prompt += f"Example {i} (Political Ideology: {ideology_names[exemplar['ideology']]}):\n"
        prompt += here_is_the_news_article + exemplar['text'] + '\n\n'
        
        # Load the pre-computed simple response for the exemplar
        response = load_exemplar_response(exemplar['name'], exemplar['dataset'], 'simple')
        if response:
            prompt += response.strip() + '\n\n'
        else:
            prompt += "SUMMARY:\n[Summary not found]\nEND OF SUMMARY.\n\n"
            
    prompt += "Now, here is the target news article to summarize:\n"
    prompt += here_is_the_news_article + article_text + '\n\n'
    prompt += closing
    
    dump_prompt(prompt, destination_folder, 'simple_fewshot')


def dump_cot_fewshot_prompt(article_text, target_article_name, dataset, destination_folder, num_shots=3):
    selector = get_exemplar_selector()
    triplet = selector.select_exemplars(article_text, target_article_name)
    
    prompt = moral_intro + '\n\n'
    prompt += "Please study the following examples of news articles, their moral word extraction, and summaries preserving the moral framing:\n\n"
    
    ideology_names = { -1: "Left", 0: "Center", 1: "Right" }
    
    for i, exemplar in enumerate(triplet, 1):
        prompt += f"Example {i} (Political Ideology: {ideology_names[exemplar['ideology']]}):\n"
        prompt += here_is_the_news_article + exemplar['text'] + '\n\n'
        
        # Load the pre-computed CoT response for the exemplar
        response = load_exemplar_response(exemplar['name'], exemplar['dataset'], 'cot')
        if response:
            prompt += response.strip() + '\n\n'
        else:
            prompt += "STEP 1:\nSUMMARY:\n[Summary not found]\nEND OF SUMMARY.\n\n"
            
    prompt += "Now, here is the target news article to summarize:\n"
    prompt += here_is_the_news_article + article_text + '\n\n'
    prompt += cot_instructions + preserve_moral_words + closing
    
    dump_prompt(prompt, destination_folder, 'cot_fewshot')


def dump_prompts(article_name, dataset, predicions_df, prompt_folder='results/test_prompts', deduplicate=True, num_shots=3):
    article_path = os.path.join(EMONA_dataset_path, article_folders[dataset], f'{article_name}.txt')
    article_text = get_clean_article_text(article_path)
    article_prompt = make_article_prompt(article_text)

    destination_folder = os.path.join(prompt_folder, dataset, article_name)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    dump_vanilla_prompt(article_prompt, destination_folder)
    dump_simple_prompt(article_prompt, destination_folder)
    dump_cot_prompt(article_prompt, destination_folder)
    dump_oracle_prompt(article_prompt, destination_folder, dataset, article_name, deduplicate)
    dump_class_prompt(article_prompt, destination_folder, predicions_df, article_name, deduplicate)
    
    # Generate balanced few-shot prompts
    dump_simple_fewshot_prompt(article_text, article_name, dataset, destination_folder, num_shots)
    dump_cot_fewshot_prompt(article_text, article_name, dataset, destination_folder, num_shots)
    
    # Generate balanced, MFT-guided few-shot prompts
    dump_simple_fewshot_mft_prompt(article_text, article_name, dataset, destination_folder, num_shots)
    dump_cot_fewshot_mft_prompt(article_text, article_name, dataset, destination_folder, num_shots)


# MFT Mapping and active dimension extraction helpers
mft_foundations_mapping = {
    'care': 'Care/Harm', 'harm': 'Care/Harm',
    'fairness': 'Fairness/Cheating', 'cheating': 'Fairness/Cheating',
    'loyalty': 'Loyalty/Betrayal', 'betrayal': 'Loyalty/Betrayal',
    'authority': 'Authority/Subversion', 'subversion': 'Authority/Subversion',
    'purity': 'Sanctity/Degradation', 'degradation': 'Sanctity/Degradation'
}

def get_active_foundations_for_article(dataset, article_name):
    """
    Get the list of unique active MFT foundations for a given article based on its gold annotations.
    """
    try:
        annotations = load_annotations(article_name, dataset)
        moral_annotations = get_moral_annotations(annotations)
        foundations = set()
        for ann in moral_annotations:
            label = ann['label']
            if label in mft_foundations_mapping:
                foundations.add(mft_foundations_mapping[label])
        if not foundations:
            return ["General Morality"]
        return sorted(list(foundations))
    except Exception as e:
        print(f"Warning: Failed to extract MFT foundations for {article_name} in {dataset}: {e}")
        return ["General Morality"]


def dump_simple_fewshot_mft_prompt(article_text, target_article_name, dataset, destination_folder, num_shots=3):
    selector = get_exemplar_selector()
    triplet = selector.select_exemplars(article_text, target_article_name)
    
    prompt = "You have to summarize the following text preserving the moral framing that the author gave to it, with particular focus on the specific moral foundations highlighted in the text.\n\n"
    prompt += "Please study the following examples of news articles and their summaries preserving the moral framing:\n\n"
    
    ideology_names = { -1: "Left", 0: "Center", 1: "Right" }
    
    for i, exemplar in enumerate(triplet, 1):
        foundations = get_active_foundations_for_article(exemplar['dataset'], exemplar['name'])
        foundations_str = ", ".join(foundations)
        prompt += f"Example {i} (Political Ideology: {ideology_names[exemplar['ideology']]}, Moral Foundations: {foundations_str}):\n"
        prompt += here_is_the_news_article + exemplar['text'] + '\n\n'
        
        # Load the pre-computed simple response for the exemplar
        response = load_exemplar_response(exemplar['name'], exemplar['dataset'], 'simple')
        if response:
            prompt += response.strip() + '\n\n'
        else:
            prompt += "SUMMARY:\n[Summary not found]\nEND OF SUMMARY.\n\n"
            
    # For target article
    target_foundations = get_active_foundations_for_article(dataset, target_article_name)
    target_foundations_str = ", ".join(target_foundations)
    
    prompt += "Now, here is the target news article to summarize:\n"
    prompt += here_is_the_news_article + article_text + '\n\n'
    prompt += f"For this target article, the author's moral framing heavily relies on the following Moral Foundation dimensions: {target_foundations_str}.\n"
    prompt += f"Please pay special attention to preserving the moral framing related to these specific dimensions ({target_foundations_str}) in your summary.\n\n"
    prompt += closing
    
    dump_prompt(prompt, destination_folder, 'simple_fewshot_mft')


def dump_cot_fewshot_mft_prompt(article_text, target_article_name, dataset, destination_folder, num_shots=3):
    selector = get_exemplar_selector()
    triplet = selector.select_exemplars(article_text, target_article_name)
    
    prompt = "You have to summarize the following text preserving the moral framing that the author gave to it, with particular focus on the specific moral foundations highlighted in the text.\n\n"
    prompt += "Please study the following examples of news articles, their moral word extraction, and summaries preserving the moral framing:\n\n"
    
    ideology_names = { -1: "Left", 0: "Center", 1: "Right" }
    
    for i, exemplar in enumerate(triplet, 1):
        foundations = get_active_foundations_for_article(exemplar['dataset'], exemplar['name'])
        foundations_str = ", ".join(foundations)
        prompt += f"Example {i} (Political Ideology: {ideology_names[exemplar['ideology']]}, Moral Foundations: {foundations_str}):\n"
        prompt += here_is_the_news_article + exemplar['text'] + '\n\n'
        
        # Load the pre-computed CoT response for the exemplar
        response = load_exemplar_response(exemplar['name'], exemplar['dataset'], 'cot')
        if response:
            prompt += response.strip() + '\n\n'
        else:
            prompt += "STEP 1:\nSUMMARY:\n[Summary not found]\nEND OF SUMMARY.\n\n"
            
    # For target article
    target_foundations = get_active_foundations_for_article(dataset, target_article_name)
    target_foundations_str = ", ".join(target_foundations)
    
    prompt += "Now, here is the target news article to summarize:\n"
    prompt += here_is_the_news_article + article_text + '\n\n'
    
    mft_cot_instructions = f"(1) First, you identify all the single words that are morally framed, focusing especially on the dimensions of {target_foundations_str}. Identify this step as \"STEP 1:\" and report each word in a new line starting with *\n" \
                           f"(2) Finally, you write a summary of the news article. Please preserve as many morally framed words as possible, particularly focusing on the moral dimensions of {target_foundations_str} in your summary. "
    
    prompt += mft_cot_instructions + closing
    
    dump_prompt(prompt, destination_folder, 'cot_fewshot_mft')

