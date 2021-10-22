from .configs import GlueConfig

import numpy as np
from typing import Union
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from newlm.lm.elmo.modeling_elmo.elmo_for_classification import (
    ELMOGPTForSequenceClassification,
)
from transformers import GPT2Config
from datasets import load_dataset, load_metric
from loguru import logger
import wandb

from transformers import BertTokenizerFast


class ClsTrainer:
    def __init__(
        self,
        pretrained_model: str,
        pretrained_tokenizer: str,
        from_scratch: str = False,
        model_config: dict = None,
        max_len: int = 512,
        model_type: str = "bert",
    ):
        self.pretrained_model = pretrained_model
        self.pretrained_tokenizer = pretrained_tokenizer
        self.from_scratch = from_scratch
        self.model_config = model_config
        self.max_len = max_len
        self.model_type = model_type

        if model_type == "elmo-gpt":
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.pretrained_tokenizer
            )
        elif model_type == "gpt2":
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.pretrained_tokenizer
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_tokenizer,
                # max_len=self.max_len,
                # truncation=True,
                use_fast=True,
            )

    def train_and_eval(self, task: str, output_dir: str, training_args: dict):
        """
        Train and Eval GLUE dataset

        Parameters
        ----------
        task : str
            Existing GLUE Task
        output_dir : str
            Path to output dir
        training_args : dict
            Training params based on transformers.TrainingArguments
        """

        glue_config = GlueConfig(task)
        dataset = self._get_dataset(glue_config)
        metric = self._get_metric(glue_config)
        model = self._get_model(glue_config.num_labels)

        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            metric_for_best_model=glue_config.metric_name,
            **training_args,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if self.model_type == "elmo-gpt":
                predictions = predictions[
                    0
                ]  # it has tuple, we need to access the index 0 for its prediction
            if task != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset[glue_config.training_key],
            eval_dataset=dataset[glue_config.validation_key],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        result = trainer.evaluate()
        trainer.save_metrics("all", result)

        wandb.finish()

    def _get_model(self, num_labels):
        if self.model_type == "bert" or self.model_type == "bert-causal":
            model = self._get_bert_model(num_labels)
        elif self.model_type == "elmo-gpt":
            model = self._get_elmo_model(num_labels)
        elif self.model_type == 'gpt2':
            model = self._get_gpt_model(num_labels)
        else:
            NotImplementedError(f"{self.model_type} is not implemented!")
        logger.info(f"Use model {type(model)}")
        return model

    def _get_gpt_model(self, num_labels):
        """
        Get GPT2 Model!
        """
        if self.from_scratch:
            cfg = GPT2Config(
                **self.model_config,
                pad_token_id=self.tokenizer.pad_token_id,
                num_labels=num_labels,
            )
            model = GPT2ForSequenceClassification(cfg)
        else:
            model = GPT2ForSequenceClassification.from_pretrained(
                self.pretrained_model, num_labels=num_labels
            )
        return model

    def _get_elmo_model(self, num_labels):
        """
        Get ELMO Model!
        """
        if self.from_scratch:
            cfg = GPT2Config(
                **self.model_config,
                pad_token_id=self.tokenizer.pad_token_id,
                num_labels=num_labels,
            )
            model = ELMOGPTForSequenceClassification(cfg)
        else:
            model = ELMOGPTForSequenceClassification.from_pretrained(
                self.pretrained_model, num_labels=num_labels
            )
        return model

    def _get_bert_model(self, num_labels):
        if self.from_scratch:
            cfg = BertConfig(
                **self.model_config,
                num_labels=num_labels,
            )
            model = BertForSequenceClassification(cfg)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model,
                num_labels=num_labels,
            )
        return model

    def _get_metric(self, glue_config: GlueConfig):
        return load_metric("glue", glue_config.actual_task)

    def _get_dataset(self, glue_config: GlueConfig):
        dataset = load_dataset("glue", glue_config.actual_task)
        sentence1_key, sentence2_key = glue_config.keys

        def preprocess_function(examples):
            if sentence2_key is None:
                return self.tokenizer(examples[sentence1_key], truncation=True)
            return self.tokenizer(
                examples[sentence1_key], examples[sentence2_key], truncation=True
            )

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        return encoded_dataset
