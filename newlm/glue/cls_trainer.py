from .configs import GlueConfig

import numpy as np
from typing import Union
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_metric


class ClsTrainer:
    def __init__(
        self,
        pretrained_model: str,
        pretrained_tokenizer: str,
        max_len: int = 512,
    ):
        self.pretrained_model = pretrained_model
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_len = max_len

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
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            num_labels=glue_config.num_labels,
        )

        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            metric_for_best_model=glue_config.metric_name,
            **training_args
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
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
        trainer.evaluate()

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
