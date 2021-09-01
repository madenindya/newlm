import torch
import os
from pathlib import Path
from typing import Union
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForPreTraining,
    BertTokenizerFast,
    PreTrainedTokenizer,
    TextDataset,
    LineByLineTextDataset,
    TextDatasetForNextSentencePrediction,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from ...utils.file_util import create_dir
import wandb
from loguru import logger

# TODO:
# - take out data from this class then pass it only on training


class LMBuilder:
    """
    Wrapper class to train BERT LM. Here, we utilize HuggingFace Trainer to train the model.
    You only need to define your tokenizer and training data, then it would train from scratch.
    """

    def __init__(
        self,
        model_config,
        tokenizer: Union[str, PreTrainedTokenizer],
        max_len: int = 512,
    ):
        self.max_len = max_len
        self.model_config = model_config
        self.tokenizer = tokenizer
        if type(tokenizer) == str:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                tokenizer,
                max_len=self.max_len,
                do_lower_case=False,  # uncased
            )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

    def create(
        self,
        train_path: str,
        output_dir: str,
        training_args: dict,
        use_nsp: bool = False,
        train_params={},
    ):
        """
        Train BERT MLM (and NSP (optional)) from scratch.

        Parameters
        ----------
        train_path : str
            Path to training file
        output_dir : str
            Path to output dir
        training_args : dict
            Training params based on transformers.TrainingArguments
        use_nsp : bool
            Wether to train NSP too or not, default: True
        """
        config = BertConfig(**self.model_config)
        if use_nsp:
            dataset = self.__get_dataset_nsp(train_path)
            model = BertForPreTraining(config=config)
        else:
            dataset = self.__get_dataset(train_path)
            model = BertForMaskedLM(config=config)

        create_dir(output_dir)
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            **training_args,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )

        self.__resolve_checkpoint(train_params, output_dir)
        if "resume_from_checkpoint" in train_params:
            logger.info(
                f"Resume training from checkpoint {train_params['resume_from_checkpoint']}"
            )
        trainer.train(**train_params)
        trainer.save_model(output_dir)

        wandb.finish()

    def __get_dataset(self, train_path):
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=self.max_len,
        )
        logger.info("Constructing roBERTa style dataset")

        # merge multiple lines to form a single example
        merged_dataset = []
        for d in dataset:
            d = d["input_ids"]
            d_len = len(d) - 2  # exclude CLS and SEP
            if (
                len(merged_dataset) > 0
                and merged_dataset[-1].size()[0] + d_len < self.max_len
            ):
                merged_dataset[-1] = torch.cat((merged_dataset[-1][:-1], d[1:]), dim=0)
            else:
                merged_dataset.append(d)

        merged_dataset = [{"input_ids": d} for d in merged_dataset]

        return merged_dataset

    def __get_dataset_nsp(self, train_path):
        return TextDatasetForNextSentencePrediction(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=self.max_len,
            nsp_probability=0.5,
        )

    def __resolve_checkpoint(self, train_params, output_dir):
        resume_from = train_params.get("resume_from_checkpoint")
        if resume_from == "latest":
            latest_ckpt = ""
            max_ckpt = 0
            for d in os.listdir(output_dir):
                if "checkpoint" in d:
                    ckpt = int(d.split("checkpoint-")[1])
                    if ckpt > max_ckpt:
                        max_ckpt = ckpt
                        latest_ckpt = str(Path(output_dir) / d)
            train_params["resume_from_checkpoint"] = (
                latest_ckpt if max_ckpt > 0 else output_dir
            )
