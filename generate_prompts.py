import os
import argparse

from moral_summarization.data_utils import EMONA_datasets, EMONA_dataset_path, article_folders
from moral_summarization.prompts import dump_prompts, load_predictions_df


parser = argparse.ArgumentParser()
parser.add_argument('--prompt-dir',       type=str,  default="results/test_prompts")
parser.add_argument('--predictions-path', type=str,  default="results/predictions_with_words.csv")
parser.add_argument('--deduplicate',      type=bool, action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--num-shots',        type=int,  default=3, help='Number of few-shot examples to inject (default: 3)')

args = parser.parse_args()

predictions_df = load_predictions_df(args.predictions_path)

for dataset in EMONA_datasets:
    dataset_path = os.path.join(EMONA_dataset_path, article_folders[dataset])
    article_names = [os.path.splitext(f)[0] for f in os.listdir(dataset_path)]
    for article_name in article_names:
        dump_prompts(article_name, dataset, predictions_df, args.prompt_dir, args.deduplicate, args.num_shots)