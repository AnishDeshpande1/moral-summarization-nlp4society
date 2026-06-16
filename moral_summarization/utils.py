import yaml
import json
import torch
import gc
from collections import Counter
import math
import numpy as np
from scipy.spatial.distance import jensenshannon
from nltk.corpus import wordnet31 as wn31
import string
from transformers import AutoTokenizer
import pickle


positive_labels = ['care', 'fairness', 'loyalty', 'authority', 'purity']
negative_labels = ['harm', 'cheating', 'betrayal', 'subversion', 'degradation']
moral_labels = positive_labels + negative_labels
all_labels = moral_labels + ['non-moral', 'non-event']


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        dict = pickle.load(file)
    return dict


def dump_pickle(file_path, dict):
    with open(file_path, 'wb') as file:
        pickle.dump(dict, file)


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        dict = yaml.safe_load(file)
    return dict


def dump_yaml(file_path, dict):
    with open(file_path, 'w') as file:
        yaml.dump(dict, file, default_flow_style=False)


def load_json(file_path):
    with open(file_path, 'r') as file:
        dict = json.load(file)
    return dict


def dump_json(file_path, dict):
    with open(file_path, 'w') as file:
        json.dump(dict, file)


def write_to_file(file_path, text):
    with open(file_path, 'w') as file:
        file.truncate(0)
        file.write(text)


def read_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def get_device_map():
    if torch.cuda.is_available():
        print(f"Using GPU {torch.cuda.get_device_name(0)}")
        device_map = {"": 0} # Use GPU 0
        device_type = "cuda"
    else:
        print('No GPU available, using the CPU instead.')
        device_map = None
        device_type = "cpu"
    return device_map, device_type


def print_vram_info():
    free_memory, total_memory = torch.cuda.mem_get_info()

    free_memory_gb = free_memory / 1024**3
    total_memory_gb = total_memory / 1024**3

    print(f"Free Memory: {free_memory_gb:.2f}/{total_memory_gb:.2f} GB")


def clear_vram(variable=None):
    if variable != None:
        del variable
    gc.collect()
    torch.cuda.empty_cache()


def check_bf16_compatibility(config):
    if config['bnb_4bit_compute_dtype'] == torch.bfloat16 and config['load_in_4bit']:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("="*80)
            print("Your GPU supports bfloat16, you are getting accelerate training with bf16= True")
            print("="*80)


def compute_normalized_entropy(annotations, num_labels):
    """
    Compute the normalized entropy of a list of annotations, example:
    annotations = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    num_labels = 3
    normalized_entropy = 1.0
    :param annotations: a list of raw annotations for one item. Each element represents the label of one annotator.
    :param num_labels: how many classes are possible?
    :return: normalized entropy, a float between 0 and 1. A higher value means more disagreement.
    """
    # frequency for each label
    data_count = Counter(annotations)
    # number of annotations
    total_count = len(annotations)
    entropy = 0.0

    for count in data_count.values():
        # probability of each label
        probability = count / total_count
        # entropy of each label
        entropy -= probability * math.log2(probability)
    # normalized entropy to the number of labels, normalized entropy is between 0 and 1.
    normalized_entropy = entropy / math.log2(num_labels)
    return normalized_entropy


def get_distribution(moral_annotations):
    """
    Get the distribution of moral labels from a list of moral annotations.
    """
    # Make a dict with moral_labels as keys and 0 as values
    moral_labels_dict = {label: 0 for label in moral_labels}

    if len(moral_annotations) == 0:
        return moral_labels_dict

    # Count the occurrences of each label
    for annotation in moral_annotations:
        label = annotation['label']
        moral_labels_dict[label] += 1

    # Normalize the counts to get a distribution
    total = sum(moral_labels_dict.values())
    distribution = {label: count / total for label, count in moral_labels_dict.items()}

    return distribution


def jsdiv_moral_annotations(annotations1, annotations2):
    """
    Compute the Jensen-Shannon divergence between two sets of moral annotations.
    """
    distribution1 = list(get_distribution(annotations1).values())
    distribution2 = list(get_distribution(annotations2).values())

    # Check if distribution1 or 2 is only made of 0s (i.e. no moral annotations)
    if sum(distribution1) == 0 or sum(distribution2) == 0:
        return np.nan
    else:
        return jensenshannon(distribution1, distribution2)
    

def remove_punctuation(word):
    punctuation = string.punctuation +'“”' # some strange characters that are not in the string.punctuation
    if word in punctuation:
        return None
    return word.lower().translate(str.maketrans('', '', punctuation)) # remove punctuation


def lemmatize(word):
    if wn31.morphy(word):
        return wn31.morphy(word)
    else:
        return word


def lemmatize_and_clean(word):
    word = remove_punctuation(word)
    if word == None:
        return None
    return lemmatize(word)


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_text(text):
    try:
        import torchtext
        from torchtext.data import get_tokenizer as get_torchtext_tokenizer
        tokenizer = get_torchtext_tokenizer("basic_english")
        return tokenizer(text)
    except ModuleNotFoundError:
        import re
        text = text.lower()
        text = re.sub(r'([.,!?()"[\]:;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()


def get_text_length(text):
    return len(tokenize_text(text))
