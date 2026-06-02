from tqdm import tqdm
from sklearn.metrics import f1_score

from .data_utils import *


def make_stats_dataframe(extra_columns=[]):
    df = pd.DataFrame(columns=prompt_types + extra_columns)
    df.index.name = 'article'
    return df


def initialize_article_stats(models, metrics, prompt_types, article_text, moral_annotations):
    article_stats = {}
    for model in models:
        article_stats[model] = {}
        for metric in metrics.keys():
            article_stats[model][metric] = dict.fromkeys(prompt_types, 0)

        # Initialize the count for the article
        article_stats[model]['length']['original'] = get_text_length(article_text)

        # Initialize the storage of the count of moral words in the article
        moral_words = [lemmatize_and_clean(annotation['token']) for annotation in moral_annotations]
        moral_words = list(dict.fromkeys(moral_words)) # remove duplicates
        article_stats[model]['moral_count']['original'] = len(moral_words)

        # Initialize the storage of the actual moral words in the article (logged but not used for stats)
        article_stats[model]['moral_words'] = dict.fromkeys(prompt_types, 0)
        article_stats[model]['moral_words']['original'] = moral_words

    return article_stats


def add_mean_and_std(df, datasets):
    df.loc[('mean', 'all'), :] = df.loc[datasets].mean(skipna=True)
    df.loc[('std', 'all'), :] = df.loc[datasets].std(skipna=True)
    for dataset in datasets:
        df.loc[('mean', dataset), :] = df.loc[dataset].mean(skipna=True)
        df.loc[('std', dataset), :] = df.loc[dataset].std(skipna=True)

    df = df.loc[datasets + ['mean', 'std'], :].round(2) # reorder the indices

    return df


class Evaluator:
    def __init__(
            self,
            models=['Meta-Llama-3-70B-Instruct'],
            extra_metrics=[], # 'summaC', 'BLANC', 'QaFactEval'
            seed=None,
            only_test_set=False,
            device='cuda',
            ):
        self.models = models
        
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
                
        self.device = device
        self.only_test = only_test_set
        self.seed = seed
        self.metrics = {'length':{}, 'moral_count':{}, 'moral_div':{}}
        self.extra_metrics = extra_metrics
        self.eval_models = {}

        # Initalize the extra metrics (reference-free automated metrics)
        for metric in self.extra_metrics:
            self.metrics[metric] = {}
            self.initialize_metric_model(metric)

        # Initialize the dictionaries to store the results
        for metric in self.metrics.keys():
            for model in self.models:
                self.metrics[metric][model] = {}

    def initialize_metric_model(self, metric):
        if metric == 'summaC':
            from summac.model_summac import SummaCConv
            self.eval_models[metric] = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=self.device, start_file="default", agg="mean")
        elif metric == 'BLANC':
            from blanc import BlancHelp
            self.eval_models[metric] = BlancHelp(device=self.device)
        elif metric == 'QaFactEval':
            from qafacteval import QAFactEval
            if self.device != "cuda":
                raise ValueError("QAFactEval only supports cuda device")
            kwargs = {"cuda_device": 0, "use_lerc_quip": True, \
                    "verbose": True, "generation_batch_size": 32, \
                    "answering_batch_size": 32, "lerc_batch_size": 8}
            model_folder = "/home/eliscio/dev/QAFactEval/models"
            self.eval_models[metric] = QAFactEval(
                lerc_quip_path=f"{model_folder}/quip-512-mocha",
                generation_model_path=f"{model_folder}/generation/model.tar.gz",
                answering_model_dir=f"{model_folder}/answering",
                lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
                lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
                **kwargs
            )
        
    def run_extra_metric_model(self, metric, article, summary):
        if metric == 'summaC':
            return self.eval_models[metric].score([article], [summary])['scores'][0]
        elif metric == 'BLANC':
            return self.eval_models[metric].eval_once(article, summary)
        elif metric == 'QaFactEval':
            results = self.eval_models[metric].score_batch_qafacteval([article], [[summary]], return_qa_pairs=True)
            return results[0][0]['qa-eval']['lerc_quip']
        
    def initialize_dataset_stats(self, dataset):
        for metric in self.metrics.keys():
            for model in self.models:
                if metric in ['length', 'moral_count']:
                    self.metrics[metric][model][dataset] = make_stats_dataframe(extra_columns=['original'])
                else:
                    self.metrics[metric][model][dataset] = make_stats_dataframe()

    def fill_article_stats(self, article_stats, prompt_type, article, summaries, moral_annotations):
        # Initialize all the article_stats fields with an empty list
        for metric in self.metrics.keys():
            article_stats[metric][prompt_type] = []
        article_stats['moral_words'][prompt_type] = []

        for summary in summaries:
            # Log the length of the response
            article_stats['length'][prompt_type].append(get_text_length(summary))

            # Get and count the moral annotations that are in the response
            num_words, moral_count = count_moral_words(summary, moral_annotations)
            article_stats['moral_count'][prompt_type].append(num_words)
            article_stats['moral_words'][prompt_type].append(
                [annotation['token'] for annotation in moral_count])

            # Compute the JS divergence between the distribution of moral labels in the article and the summary
            moral_div = jsdiv_moral_annotations(moral_annotations, moral_count)
            article_stats['moral_div'][prompt_type].append(round(moral_div, 3))

            # Run the extra metric models
            for metric in self.extra_metrics:
                article_stats[metric][prompt_type].append(
                    self.run_extra_metric_model(metric, article, summary))

        for metric in self.metrics.keys():
            article_stats[metric][prompt_type] = np.mean(article_stats[metric][prompt_type])

        return article_stats

    def get_results_for_one_article(self, results_article_path, article, dataset, return_df=False):
        # Load the article text
        article_text = load_article(article, dataset)

        # Load the annotations for the article
        annotations = load_annotations(article, dataset)
        moral_annotations = get_moral_annotations(annotations)

        article_stats = initialize_article_stats(
            self.models, self.metrics, prompt_types, article_text, moral_annotations)

        # Get all files with responses
        responses_files = [file_path for file_path in os.listdir(results_article_path)\
                           if 'response' in file_path]
        
        # if there are no response_files containing 'class' and self.only_test=True, skip
        if self.only_test and len([file_path for file_path in responses_files if 'class' in file_path]) == 0:
            return

        for model in self.models:
            for prompt_type in prompt_types:
                # get all the files with this model and this prompt style
                response_files = [file_path for file_path in responses_files\
                                  if model in file_path and file_path.startswith(f"{prompt_type}_response_")]
                
                # only articles in the test set have a class summary
                if len(response_files) == 0:
                    for metric in self.metrics.keys():
                        article_stats[model][metric][prompt_type] = np.nan
                    continue

                if self.seed:
                    response_files = [file_path for file_path in response_files if self.seed in file_path]

                summaries = []
                for file_path in response_files:
                    response_path = os.path.join(results_article_path, file_path)
                    summary_text = get_summary_from_response(response_path)
                    if summary_text is not None:
                        summaries.append(summary_text)

                article_stats[model] = self.fill_article_stats(
                    article_stats[model], prompt_type, article_text, summaries, moral_annotations)

        # Store the stats for the article
        for metric in self.metrics.keys():
            for model in self.models:
                self.metrics[metric][model][dataset].loc[article] = article_stats[model][metric]

        if return_df:
            for model in self.models:
                article_stats[model] = pd.DataFrame(article_stats[model])
            return pd.concat(article_stats, names=['model'])

    def postprocess_evaluation_results(self):
        # convert dicts to multi-index dataframe
        for metric in self.metrics.keys():
            for model in self.models:
                self.metrics[metric][model] = pd.concat(self.metrics[metric][model], names=['dataset'])

                # add mean and std rows
                datasets = self.metrics[metric][model].index.get_level_values('dataset').unique().to_list()
                self.metrics[metric][model] = add_mean_and_std(self.metrics[metric][model], datasets)

            self.metrics[metric] = pd.concat(self.metrics[metric], names=['model'])

    def evaluate_summaries(self, results_dir):
        # loop through datasets (allsides, basil, mpqa)
        for dataset in os.listdir(results_dir):
            dataset_path = os.path.join(results_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue

            self.initialize_dataset_stats(dataset)

            for article in tqdm(os.listdir(dataset_path)):
                # loop through articles in each dataset
                article_path = os.path.join(dataset_path, article)
                if not os.path.isdir(article_path):
                    continue

                # get the summary info of the results for the article
                self.get_results_for_one_article(article_path, article, dataset)

        return self.postprocess_evaluation_results()


def f1_moral_predictions(moral_words, predictions):
    vocabulary = list(set(moral_words + predictions))  # Unique words
    y_true = [1 if word in moral_words else 0 for word in vocabulary]
    y_pred = [1 if word in predictions else 0 for word in vocabulary]

    return f1_score(y_true, y_pred)


def evaluate_CoT_moral_words_predictions(results_dir, models, only_test_set=False, article_list=None, seed=None):
    if only_test_set is False:
        article_list = [os.listdir(os.path.join(results_dir, dataset)) for dataset in os.listdir(results_dir)]
        article_list = [article for dataset in article_list for article in dataset]
        results_df = pd.DataFrame(columns=models + ['is_in_test_set'], index=article_list)
        length_df = pd.DataFrame(columns=models + ['is_in_test_set'], index=article_list)
    elif only_test_set and article_list:
        with open(article_list) as f:
            article_list = f.read().splitlines()
            results_df = pd.DataFrame(columns=models, index=article_list)
            length_df = pd.DataFrame(columns=models, index=article_list)
    else:
        raise ValueError("If only_test_set is True, article_list must be provided")

    for dataset in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset)

        for article in tqdm(os.listdir(dataset_path)):
            if article not in article_list:
                continue

            # loop through articles in each dataset
            article_path = os.path.join(dataset_path, article)

            # # check if the article is in the test set
            if only_test_set is False:
                classification_files = [file_path for file_path in os.listdir(article_path)\
                            if 'class' in file_path]
                if len(classification_files) > 0:
                    results_df.loc[article, 'is_in_test_set'] = True
                else:
                    results_df.loc[article, 'is_in_test_set'] = False

            annotations = load_annotations(article, dataset)
            moral_annotations = get_moral_annotations(annotations)
            moral_words = [lemmatize_and_clean(annotation['token']) for annotation in moral_annotations]
            moral_words = list(dict.fromkeys(moral_words)) # remove duplicates

            # Get all files with responses
            responses_files = [file_path for file_path in os.listdir(article_path)\
                            if 'response' in file_path]

            for model in models:
                response_files = [file_path for file_path in responses_files\
                    if model in file_path and 'cot' in file_path]

                if seed:
                    response_files = [file_path for file_path in response_files if seed in file_path]

                f1_seeds = []
                length_predictions = []
                for file_path in response_files:
                    response_path = os.path.join(article_path, file_path)

                    predictions = get_cot_prediction_from_response(response_path)
                    length_predictions.append(len(predictions))
                    if predictions is None or len(predictions) == 0:
                        continue

                    f1_seeds.append(f1_moral_predictions(moral_words, predictions))

                results_df.loc[article, model] = np.mean(f1_seeds)
                length_df.loc[article, model] = np.mean(length_predictions)

    # Add mean to the dataframe
    results_df.loc['mean'] = results_df.mean(axis=0)
    length_df.loc['mean'] = length_df.mean(axis=0)

    # add mean only for rows where is_in_test_set is True
    if only_test_set is False:
        results_df.loc['mean_test_set'] = results_df[results_df['is_in_test_set'] == True].mean(axis=0)

    return results_df, length_df