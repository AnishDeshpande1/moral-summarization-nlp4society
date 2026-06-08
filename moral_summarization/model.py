import time
import os
from tqdm import tqdm
import torch
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
    )
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model
    )
from trl import SFTTrainer, SFTConfig
from safetensors.torch import save_file, load_file

from .utils import (
    get_device_map,
    clear_vram,
    check_bf16_compatibility,
    get_tokenizer,
    dump_json
    )
from .metrics import (
    sequence_classification_metrics,
    token_classification_metrics,
    evaluate_on_df
    )
from .data_utils import process_tokenizer_results_df


def load_classifier_head(model, classifier_path):
    cls_head_state = load_file(classifier_path)
    model.score.weight.data.copy_(cls_head_state['model.score'])
    return model


def merge_adapters_to_base_model(model, adapters_path, classifier_path):
    model = PeftModel.from_pretrained(model, adapters_path)
    tokenizer = get_tokenizer(adapters_path)

    # Load classification head
    model = load_classifier_head(model, classifier_path)
    return model, tokenizer


class LlamaModelForClassification:
    def __init__(self, config):
        self.config = config
        self.verbose = config['verbose']
        self.num_labels = config['num_labels']
        self.base_model_name = config['base_model_name']
        self.base_model_path = os.path.join(config['hf_model_folder'], self.base_model_name)

        # Get BitsAndBytes configuration
        self.get_bnb_config()

        # Load base model
        self.load_base_model()

        # Prepare PEFT model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(**self.config['lora']))

        # Load tokenizer
        self.tokenizer = get_tokenizer(self.base_model_path)

        if 'pretrained_model_name' in self.config:
            self.load_pretrained_model()

        # Update some model configs
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.gradient_checkpointing_enable({'use_reentrant': False})

    def get_bnb_config(self):
        if 'bitsandbytes' not in self.config:
            raise ValueError("BitsAndBytes configuration is required for this model.")

        # check if the compute type is a string
        if isinstance(self.config['bitsandbytes']['bnb_4bit_compute_dtype'], str):
            self.config['bitsandbytes']['bnb_4bit_compute_dtype'] = getattr(
                torch, self.config['bitsandbytes']['bnb_4bit_compute_dtype'])

        self.bnb_config = BitsAndBytesConfig(**self.config['bitsandbytes'])
        self.dtype = self.bnb_config.bnb_4bit_compute_dtype

        # Check if we can load the model on a GPU
        self.device_map, self.device_type = get_device_map()
        if self.device_type == 'cuda' and 'bitsandbytes' in self.config and self.verbose:
            check_bf16_compatibility(self.config['bitsandbytes'])

    def load_base_model(self):
        raise NotImplementedError("This method should be implemented in the child class.")

    def load_pretrained_model(self):
        self.pretrained_model_name = self.config['pretrained_model_name']
        self.pretrained_model_path = os.path.join(self.config['hf_model_folder'], self.pretrained_model_name)
        if 'pretrained_classifier_path' not in self.config:
            self.config['pretrained_classifier_path'] = os.path.join(self.pretrained_model_path, 'score.safetensors')

        self.model, self.tokenizer = merge_adapters_to_base_model(
            self.model,
            self.pretrained_model_path,
            self.config['pretrained_classifier_path']
        )

    def get_training_metrics(self):
        raise NotImplementedError("This method should be implemented in the child class.")

    def train(self, dataset):
         # Create Torch trainer
        self.trainer = Trainer(
            model = self.model,
            train_dataset = dataset['train'],
            eval_dataset = dataset['val'],
            args = TrainingArguments(**self.config['training']),
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = self.get_training_metrics(),
        )

        print("Training...")
        start_time = time.time()

        self.trainer.train()

        end_time = time.time()
        execution_time = end_time - start_time
        if self.verbose:
            print("Training time:", execution_time)

    def save_model(self):
        print("Saving model...")
        save_path = self.config['training']['output_dir']
        self.trainer.save_model(save_path)

        # save the classification head
        cls_state_dict = {'model.score': self.model.score.weight.detach().clone().cpu()}
        save_file(cls_state_dict, os.path.join(save_path, 'score.safetensors'))

        self.tokenizer.save_pretrained(save_path)

    def predict_logits_on_batch(self, sentences, batch_size=1, is_split_into_words=False, max_length=4096):
        if batch_size is None:
            batch_size = self.config['training']['per_device_eval_batch_size']

        predicted_logits = []

        # Process the sentences in batches
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i+batch_size]

            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                is_split_into_words=is_split_into_words,
                max_length=max_length
            )

            # Move tensors to the device where the model is (e.g., GPU or CPU)
            inputs = {k: v.to(self.device_type) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_logits.append(outputs['logits'])

        return predicted_logits

    def predict(self, sentences, batch_size=1, is_split_into_words=False, max_length=4096, get_labels=False):
        """Perform inference on a list of sentences.
        """
        predicted_logits = self.predict_logits_on_batch(
            sentences,
            batch_size,
            is_split_into_words,
            max_length
            )

        # Process logits based on the task
        return self.process_predicted_logits(predicted_logits, get_labels)

    def process_predicted_logits(self, logits):
        raise NotImplementedError("This method should be implemented in the child class.")

    def predict_on_dataset(self, dataset, max_length=4096, predictions_file='predictions.csv', metrics_file='metrics_results.json'):
        dataset_df = dataset.to_pandas()

        # Add results to df based on the task
        dataset_df = self.add_results_to_df(dataset_df, max_length)

        # Dump results to a CSV file
        save_path = os.path.join(self.config['training']['output_dir'], predictions_file)
        dataset_df.to_csv(save_path, index=False)

        # Calculate metrics on the dataset
        metrics_results = evaluate_on_df(dataset_df, self.config['task'])

        # Dump results to a JSON file
        save_path = os.path.join(self.config['training']['output_dir'], metrics_file)
        dump_json(save_path, metrics_results)

    def add_results_to_df(self, dataset_df, max_length=4096):
        raise NotImplementedError("This method should be implemented in the child class.")


class LlamaModelForTokenClassification(LlamaModelForClassification):
    def __init__(self, config):
        self.id2label = {0: "non-moral", 1: "moral"}
        self.label2id = {"non-moral": 0, "moral": 1}

        super().__init__(config)

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def load_base_model(self):
        # Load base model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.base_model_path,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def get_training_metrics(self):
        return token_classification_metrics

    def process_predicted_logits(self, logits, get_labels=False):
        # Remove the first dimension if it is of size 1
        logits = [logit.squeeze(0) for logit in logits]

        # Apply argmax to each sentence's logits
        predictions = [logit.argmax(axis=1).cpu().numpy() for logit in logits]
        if get_labels:
            return [
                [self.id2label[t] for t in prediction]
                for prediction in predictions
            ]
        else:
            return predictions

    def add_results_to_df(self, dataset_df, max_length=4096):
        dataset_df['predictions'] = self.predict(
            dataset_df['tokens'].apply(list).to_list(), # convert to list of lists
            is_split_into_words=True,
            max_length=max_length
            )

        # Get the predicted and labeled words from the predicted and labeled tokens
        dataset_df = process_tokenizer_results_df(dataset_df, self.tokenizer)

        return dataset_df


class LlamaModelForSequenceClassification(LlamaModelForClassification):
    def __init__(self, config):
        self.id2label = {0: "non-moral", 1: "moral"}
        self.label2id = {"non-moral": 0, "moral": 1}

        super().__init__(config)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def load_base_model(self):
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path,
            quantization_config=self.bnb_config,
            device_map=self.device_map,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def get_training_metrics(self):
        return sequence_classification_metrics

    def process_predicted_logits(self, logits, get_labels=False):
        logits = torch.cat(logits, dim=0)
        predictions = logits.argmax(axis=1).cpu().numpy()
        if get_labels:
            return [self.id2label[t.item()] for t in predictions]
        else:
            return predictions

    def add_results_to_df(self, dataset_df, max_length=4096):
        dataset_df['predictions'] = self.predict(
            dataset_df['text'].to_list(),
            max_length=max_length
            )

        return dataset_df


class LlamaModelForSequenceCompletion:
    def __init__(self, config):
        self.config = config
        self.verbose = config['verbose']
        self.base_model_name = config['base_model_name']
        self.base_model_path = os.path.join(config['hf_model_folder'], self.base_model_name)
        if 'bitsandbytes' in config:
            # Convert dtype string to torch dtype if needed (YAML loads it as str)
            if isinstance(config['bitsandbytes'].get('bnb_4bit_compute_dtype'), str):
                config['bitsandbytes']['bnb_4bit_compute_dtype'] = getattr(
                    torch, config['bitsandbytes']['bnb_4bit_compute_dtype'])
            self.bnb_config = BitsAndBytesConfig(**config['bitsandbytes'])
            self.dtype = self.bnb_config.bnb_4bit_compute_dtype
        else:
            self.bnb_config = None
            self.dtype = torch.float16

        # Check if we can load the model on a GPU
        self.device_map, self.device_type = get_device_map()
        if self.device_type == "cuda" and 'bitsandbytes' in config and self.verbose:
            check_bf16_compatibility(config['bitsandbytes'])

        # Load base model
        if self.bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=self.bnb_config,
                device_map=self.device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def init_pipeline(self):
        # Create inference pipeline
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=self.dtype,
        )

        # Set terminators. <|eot_id|> is Llama-3 specific; only add it if the
        # tokenizer actually has it (DeepSeek/Qwen models use different tokens).
        eot_id = self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.terminators = [self.pipeline.tokenizer.eos_token_id]
        if eot_id != self.pipeline.tokenizer.unk_token_id:
            self.terminators.append(eot_id)
    
    def save_model(self, folder_name):
        print("Saving model...")
        save_path = os.path.join(self.config['training']['output_dir'], folder_name)
        self.model.save_pretrained(save_path)

        # Delete model from GPU to save memory
        clear_vram(self.model)

        if self.verbose:
            print("Reloading model...")
        # Reload model in the right dtype and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )

        if self.verbose:
            print("Merging model...")
        model = PeftModel.from_pretrained(base_model, save_path)
        model = model.merge_and_unload()

        # save tokenizer
        if self.verbose:
            print("Saving tokenizer...")
        
        self.tokenizer.save_pretrained(save_path)

    def train(self, dataset):
        # SFT Trainer
        self.trainer = SFTTrainer(
            model = self.model,
            train_dataset = dataset,
            peft_config = LoraConfig(**self.config['lora']),
            args = SFTConfig(**self.config['training']),
            tokenizer = self.tokenizer
        )

        print("Training...")
        start_time = time.time()

        self.trainer.train()

        end_time = time.time()
        execution_time = end_time - start_time
        if self.verbose:
            print("Training time:", execution_time)

    def get_response(
            self,
            query,
            system_content="You are a news summarizer assistant and a moral expert.",
            max_tokens=4096,
            temperature=0.6,
            top_p=0.9
            ):
        # Initialize pipeline once (lazily). Note: the attribute created by
        # init_pipeline is `self.pipeline`, so guard on that — guarding on
        # `pipe` rebuilt the HF pipeline on every call.
        if not hasattr(self, 'pipeline'):
            self.init_pipeline()

        # Prepare prompt in conversation format
        conversation = [{"role": "system", "content": system_content}]
        prompt = conversation + [{"role": "user", "content": query}]

        print("Prompting the model...")
        start_time = time.time()

        # Generate response
        with torch.autocast(self.device_type):
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        end_time = time.time()
        execution_time = end_time - start_time
        if self.verbose:
            print("Execution time:", execution_time)

        response = outputs[0]["generated_text"][len(prompt):]
        return response, prompt + [{"role": "assistant", "content": response}]