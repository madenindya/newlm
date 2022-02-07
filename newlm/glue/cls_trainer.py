from .configs import GlueConfig

import numpy as np
from typing import List, Union, Tuple
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
    ELMOBertForSequenceClassificationV2,
    ELMOBertForSequenceClassificationV3,
    ELMOGPTForSequenceClassification,
    ELMOBertForSequenceClassification,
)
from newlm.lm.bert.modeling_bert.bert_model import (
    BertModelCausalForSequenceClassification,
    BertModelCausalR2LForSequenceClassification,
)
from transformers import GPT2Config
from datasets import load_dataset, load_metric
from loguru import logger
import wandb

import torch
from transformers import BertTokenizerFast

try:
    from mosestokenizer import MosesDetokenizer
except:
    logger.warning("Unable to import MosesDetokenizer, function detokenize_moses could not be use!")

try:
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    detokenizer_tb = TreebankWordDetokenizer()
except:
    logger.warning("Unable to import TreebankWordDetokenizer, function detokenize_tb could not be use!")


class ClsTrainer:
    def __init__(
        self,
        pretrained_model: Union[str, Tuple[str, str]],
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

        if model_type in ["elmo-gpt", "gpt2", "elmo-bert-causal", "elmo-bert-causal-l2r-r2l", "elmo-bert-causal-l2r-r2l-v2"]:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.pretrained_tokenizer
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_tokenizer,
                use_fast=True,
            )

    def train_and_eval(
        self, task: str, output_dir: str, training_args: dict, oth_args: dict
    ):
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

        glue_config = GlueConfig(task, oth_args)
        detokenizer = None
        if glue_config.detokenizer is not None:
            logger.info(f"Use detokenizer {glue_config.detokenizer}")
            if glue_config.detokenizer == "moses":
                detokenizer = self.__detokenize_moses
            else:
                detokenizer = self.__detokenize_tb
        dataset = self._get_dataset(glue_config, detokenizer)
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
            if self.model_type in ["elmo-gpt", "elmo-bert-causal", "elmo-bert-causal-l2r-r2l"]:
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
        if self.model_type == "bert":
            model = self._get_bert_model(num_labels)
        elif self.model_type == "bert-causal":
            model = self._get_bert_causal_model(num_labels)
        elif self.model_type == "bert-causal-r2l":
            model = self._get_bert_causal_r2l_model(num_labels)
        elif self.model_type == "elmo-gpt":
            model = self._get_elmo_model(num_labels)
        elif self.model_type == "gpt2":
            model = self._get_gpt_model(num_labels)
        elif self.model_type == "elmo-bert-causal":
            model = self._get_elmo_bert_model(num_labels)
        elif self.model_type == "elmo-bert-causal-l2r-r2l": #v3
            model = self._get_elmo_bert_l2r_r2l_model(num_labels)
        elif self.model_type == "elmo-bert-causal-l2r-r2l-v2":
            model = self._get_elmo_bert_l2r_r2l_v2_model(num_labels)
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

    def _get_elmo_bert_model(self, num_labels):
        """
        Get ELMO Model!
        """
        if self.from_scratch:
            cfg = BertConfig(
                **self.model_config,
                num_labels=num_labels,
            )
            model = ELMOBertForSequenceClassification(cfg)
        else:
            model = ELMOBertForSequenceClassification.from_pretrained(
                self.pretrained_model, num_labels=num_labels
            )
        return model

    def _get_elmo_bert_l2r_r2l_model(self, num_labels): # v3
        """
        Get ELMO Model!
        """
        print("ELMO BERT R2L L2R V3")
        if self.from_scratch:
            raise Exception("bert-causal can not be finetune from scratch (for now)")
        else:
            model_l2r = BertModelCausalForSequenceClassification.from_pretrained(
                self.pretrained_model[0], num_labels=num_labels
            )
            model_r2l = BertModelCausalR2LForSequenceClassification.from_pretrained(
                self.pretrained_model[1], num_labels=num_labels
            )

            print("Initialize ELMO BERT with config", self.model_config)
            cfg = BertConfig(
                **self.model_config,
                num_labels=num_labels,
            )
            model = ELMOBertForSequenceClassificationV3(cfg)

            model.transformer.l2r_gpt = model_l2r.bert
            model.transformer.r2l_gpt = model_r2l.bert

        return model

    def _get_elmo_bert_l2r_r2l_v2_model(self, num_labels):
        """
        Get ELMO Model!
        """
        print("ELMO BERT R2L L2R V2")
        if self.from_scratch:
            raise Exception("bert-causal can not be finetune from scratch (for now)")
        else:
            model_l2r = BertModelCausalForSequenceClassification.from_pretrained(
                self.pretrained_model[0], num_labels=num_labels
            )
            model_r2l = BertModelCausalR2LForSequenceClassification.from_pretrained(
                self.pretrained_model[1], num_labels=num_labels
            )

            print("Initialize ELMO BERT with config", self.model_config)
            cfg = BertConfig(
                **self.model_config,
                num_labels=num_labels,
            )
            model = ELMOBertForSequenceClassificationV2(cfg)

            model.l2r_cls = model_l2r
            model.r2l_cls = model_r2l

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
                is_decoder=False,  # fine-tune using encoder
            )
        return model

    def _get_bert_causal_model(self, num_labels):
        """
        Get BERT Causal Model!
        use BertModelCausalForSequenceClassification
        expected to have config is_decoder=True
        """
        if self.from_scratch:
            raise NotImplementedError("bert-causal can not be finetune from scratch (for now)")
        else:
            model = BertModelCausalForSequenceClassification.from_pretrained(
                self.pretrained_model, num_labels=num_labels
            )
        return model

    def _get_bert_causal_r2l_model(self, num_labels):
        if self.from_scratch:
            raise NotImplementedError("bert-causal-r2l can not be finetune from scratch")
        else:
            model = BertModelCausalR2LForSequenceClassification.from_pretrained(
                self.pretrained_model, num_labels=num_labels
            )
        return model

    def _get_metric(self, glue_config: GlueConfig):
        return load_metric("glue", glue_config.actual_task)

    def _get_dataset(self, glue_config: GlueConfig, detokenizer=None):
        dataset = load_dataset("glue", glue_config.actual_task)
        sentence1_key, sentence2_key = glue_config.keys

        def preprocess_function(examples):
            if detokenizer is not None:
                examples[sentence1_key] = detokenizer(examples[sentence1_key])
                if sentence2_key is not None:
                    examples[sentence2_key] = detokenizer(examples[sentence2_key])
            if sentence2_key is None:
                return self.tokenizer(examples[sentence1_key], truncation=True)
            return self.tokenizer(
                examples[sentence1_key], examples[sentence2_key], truncation=True
            )

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        return encoded_dataset

    def __detokenize_moses(self, data: List[str]):
        """
        Parameters
        ----------
        data : List[str]
            original data that has been tokenized, separated by space.
            Ex: ['I book " promo " yesterday .', 'There 's 50 % discount .']

        Returns
        -------
        data : List[str]
            detokenized data
            Ex: ['I book "promo" yesterday.', 'There's 50% discount.']
        """
        with MosesDetokenizer("en") as detokenize:
            return [detokenize(d.split()) for d in data]

    def __detokenize_tb(self, data: List[str]):
        return [detokenizer_tb.detokenize(d.split()) for d in data]
