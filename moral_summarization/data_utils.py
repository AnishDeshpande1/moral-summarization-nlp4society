import os
import pandas as pd
from ast import literal_eval
import re

from .utils import *


# Try to use local cloned EMONA_dataset in root first, fallback to home directory path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_emona_path = os.path.join(root_dir, 'EMONA_dataset', 'EMONA')
if os.path.exists(local_emona_path):
    EMONA_dataset_path = local_emona_path
else:
    EMONA_dataset_path = os.path.join(os.environ['HOME'], 'dev/moral-summarization/EMONA_dataset/EMONA')

EMONA_datasets = ['allsides', 'basil', 'mpqa']
article_folders = {
    'allsides' : 'moral_allsides_articles',
    'basil'    : 'moral_basil_articles',
    'mpqa'     : 'moral_mpqa_articles'
}
annotation_folders = {
    'allsides' : 'moral_allsides_annotations',
    'basil'    : 'moral_basil_annotations',
    'mpqa'     : 'moral_mpqa_annotations'
}
prompt_types = ['vanilla', 'simple', 'cot', 'oracle', 'class', 'simple_fewshot', 'cot_fewshot', 'simple_fewshot_mft', 'cot_fewshot_mft']
test_set_articles_path = os.path.join(root_dir, 'results', 'articles_in_test_set.txt')


def get_article_path(article, dataset):
    return os.path.join(
        EMONA_dataset_path,
        article_folders[dataset],
        f'{article}.txt'
    )


def load_article(article, dataset):
    article_path = get_article_path(article, dataset)
    return read_from_file(article_path)


def get_annotations_path(article, dataset):
    return os.path.join(
        EMONA_dataset_path,
        annotation_folders[dataset],
        f'{article}.json'
    )


def load_annotations(article, dataset):
    annotation_path = get_annotations_path(article, dataset)
    return load_json(annotation_path)


def get_moral_annotations(annotations):
    moral_annotations = []
    for sentence_number, sentence in enumerate(annotations['sentences']):
        for token, label in zip(sentence['tokens_list'], sentence['tokens_label_list']):
            if label in moral_labels:
                moral_annotations.append({
                    'sentence_number' : sentence_number,
                    'token' : token,
                    'label' : label
                })

    return moral_annotations


def count_moral_words(summary, annotations):
    summary = [lemmatize_and_clean(token) for token in tokenize_text(summary)]
    summary = list(dict.fromkeys(summary)) # remove duplicates
    clean_annotations = []
    tokens = []
    for annotation in annotations:
        token = lemmatize_and_clean(annotation['token'])
        if token not in tokens:
            annotation['token'] = token
            clean_annotations.append(annotation)
            tokens.append(token)

    count = sum([1 for annotation in clean_annotations if annotation['token'] in summary])
    moral_words = [annotation for annotation in clean_annotations if annotation['token'] in summary]

    return count, moral_words


def load_results_df(path, literal_eval_columns=[]):
    #literal_eval_columns = ['predictions', 'labels'] + extra_literal_eval_columns
    converters = {column: literal_eval for column in literal_eval_columns}
    results = pd.read_csv(path, converters=converters)

    for column in results.columns:
        results[column] = results[column].apply(np.array)

    return results


def get_words_from_predictions(tokens, predictions, tokenizer):
    # Tokenize the input to get the correspondence between tokens and words
    tokenized_inputs = tokenizer(
        tokens.tolist(),
        truncation=True,
        max_length=4096,
        is_split_into_words=True
        )

    # COnvert prediction ids to bool for indexing
    pred_idx = predictions.astype(bool)

    # Get the word ids for the predicted tokens
    word_ids = np.array(tokenized_inputs.word_ids())[pred_idx]

    # De-duplicate the word ids (multiple tokens can correspond to the same word)
    word_ids = np.unique(word_ids[word_ids != None])

    # Get the words from the word ids
    predicted_words = [tokens[word_id] for word_id in word_ids if word_id != None]

    return predicted_words


def process_tokenizer_results_df(results_df, tokenizer):
    # Get predicted words from the predictions, matching tokens to words
    results_df['predicted_words'] = [
        get_words_from_predictions(
            row['tokens'],
            row['predictions'],
            tokenizer
            )
        for _, row in results_df.iterrows()]

    # Get labeled words from the predictions, matching tokens to words
    results_df['labeled_words'] = [
        get_words_from_predictions(
            row['tokens'],
            row['labels'],
            tokenizer
            )
        for _, row in results_df.iterrows()]

    for column in list(set(results_df.columns) - set(['dataset', 'article', 'sentence_id'])):
        results_df[column] = results_df[column].apply(list)

    return results_df


def find_summary_token(response):
    if 'SUMMARY:' in response:
        return 'SUMMARY:'
    elif 'Summary:' in response:
        return 'Summary:'
    elif 'SUMMERY:' in response:
        return 'SUMMERY:'
    elif '</think>' in response and 'STEP 1' not in response:
        return '</think>'
    else:
        return None


def find_end_of_summary_token(response):
    if 'END OF SUMMARY' in response:
        return 'END OF SUMMARY'
    elif 'End of Summary' in response:
        return 'End of Summary'
    elif 'End of SUMMARY' in response:
        return 'End of SUMMARY'
    elif 'End of summary' in response:
        return 'End of summary'
    else:
        return None


def get_summary_from_response(response_path):
    response = read_from_file(response_path)

    begin_token = find_summary_token(response)
    if begin_token is None:
        print(f'No SUMMARY token found in response {response_path}')
        return None

    summary_start = [match.start() for match in re.finditer(begin_token, response)][-1]
    summary_start += len(begin_token)

    end_token = find_end_of_summary_token(response)
    if end_token is None:
        print(f'No END OF SUMMARY token found in response {response_path}')
        summary_end = len(response)
    else:
        summary_end = [match.start() for match in re.finditer(end_token, response)][-1]

    if summary_start >= summary_end:
        print(f'Invalid summary start and end found in response {response_path}')
        summary = response[summary_end:summary_start].strip()
    else:
        summary = response[summary_start:summary_end].strip()

    if len(summary) == 0 or summary is None:
        print(f'Empty summary found in response {response_path}')
        return None

    return summary


def get_cot_prediction_from_response(response_path):
    """ # Get predictions from the response.
    The predictions are listed one line after the other, between "STEP 1" and "SUMMARY"
    """
    response = read_from_file(response_path)

    if 'STEP 1:' in response:
        begin_token = 'STEP 1:'
    else:
        print(f'No STEP 1 token found in response {response_path}')
        return None

    end_token = find_summary_token(response)
    if end_token is None:
        print(f'No SUMMARY token found in response {response_path}')
        return None

    begin_index = response.index(begin_token) + len(begin_token)
    end_index = response.index(end_token)
    predictions_text = response[begin_index:end_index]

    # parse one prediction from each line of predictions_text
    predictions = predictions_text.split('\n')
    predictions = [remove_punctuation(prediction)\
                    for prediction in predictions if prediction is not None]
    predictions = [lemmatize(prediction.strip())\
                    for prediction in predictions if prediction is not None]
    predictions = list(dict.fromkeys(predictions))

    return predictions


def get_quoted_spans(text):
    # Regex to match text enclosed by any type of quotes (straight or curly)
    quote_regex = r'(["“”])(.*?)(\1)' 
    quotes = re.findall(quote_regex, text)
    if len(quotes) == 0:
        return None
    else:
        return [quote[1] for quote in quotes]
    

def count_quotes_in_summaries(summaries_df):
    quotes_df = pd.DataFrame(index=summaries_df.index, columns=summaries_df.columns)
    moral_quotes_df = pd.DataFrame(index=summaries_df.index, columns=summaries_df.columns)

    for index, row in summaries_df.iterrows():
        for method in summaries_df.columns:
            quotes_df.loc[index, method] = []
            moral_quotes_df.loc[index, method] = []

    for index, row in summaries_df.iterrows():
        annotations = load_annotations(index, index.split('_')[0])
        moral_annotations = get_moral_annotations(annotations)

        for method in summaries_df.columns:
            quoted_spans = get_quoted_spans(row[method])
            if quoted_spans is None:
                continue

            quotes_df.loc[index, method] = quoted_spans
            for span in quoted_spans:
                count, _ = count_moral_words(span, moral_annotations)
                if count > 0:
                    moral_quotes_df.loc[index, method].append(span)

    return quotes_df, moral_quotes_df


def print_mean_max(scores_df, print_methods=['vanilla']):
    for method in print_methods:
        all_lengths = [len(v) if v is not np.nan else 0 for v in scores_df[method].values]
        print(method)
        print('Max number of highlights:', np.max(all_lengths))
        print(f'Mean number of highlights: {np.mean(all_lengths):.2f} \pm {np.std(all_lengths):.2f}')
        print("=====================================")


def get_articles_in_test_set(path=test_set_articles_path):
    with open(path, 'r') as f:
        articles = f.readlines()
    return [article.strip() for article in articles]


def distribution_quotes_in_articles(only_test_set=True):
    percentage_moral_in_quotes, percentage_quotes_in_text = [], []
    test_set = get_articles_in_test_set()

    for dataset in EMONA_datasets:
        dataset_path = os.path.join(EMONA_dataset_path, article_folders[dataset])
        article_names = [os.path.splitext(f)[0] for f in os.listdir(dataset_path)]
        for article_name in article_names:
            if only_test_set is True and article_name not in test_set:
                continue

            annotations = load_annotations(article_name, dataset)
            moral_annotations = get_moral_annotations(annotations)
            moral_words = [ann['token'] for ann in moral_annotations]

            article_text = load_article(article_name, dataset)
            article_length = get_text_length(article_text)
            quoted_spans = get_quoted_spans(article_text)
            quotes_length = get_text_length(' '.join(quoted_spans)) if quoted_spans else 0

            percentage_quotes_in_text.append(quotes_length / article_length)

            len_moral_words = len(moral_words)

            count_moral_words_spans = 0
            if quoted_spans:
                for span in quoted_spans:
                    # check if any of the moral words is in the quote
                    for word in moral_words:
                        if word in span:
                            count_moral_words_spans += 1
                            # pop the word from the list to avoid counting it multiple times
                            moral_words.remove(word)

            if len(moral_words) == 0:
                continue
            percentage_moral_in_quotes.append(count_moral_words_spans / len_moral_words)

    return percentage_moral_in_quotes, percentage_quotes_in_text
