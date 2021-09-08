import torch
import os

from tqdm import tqdm

from datasets import load_dataset
from pathlib import Path
from typing import Union
from transformers import (
    BertTokenizerFast,
    PreTrainedTokenizer,
    TextDataset,
    LineByLineTextDataset,
    TextDatasetForNextSentencePrediction,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from newlm.utils.file_util import create_dir
import wandb
from loguru import logger
from newlm.lm.elmo.modeling_elmo.elmo_head import ELMOGPTLMHeadModel
from transformers import GPT2Config
# TODO:
# - take out data from this class then pass it only on training


class ELMOLMBuilder:
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
            mlm=False,
        )

    def create(
        self,
        train_path: str,
        output_dir: str,
        training_args: dict,
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
        config = GPT2Config(**self.model_config)
        dataset = self.__get_dataset(train_path)
        model = ELMOGPTLMHeadModel(config=config)

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

    def __get_dataset_via_ds(self, train_path):
        dataset = load_dataset("text", data_files=train_path)

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        return encoded_dataset["train"]

    def __get_dataset(self, train_path):
        dataset = self.__get_dataset_via_ds(train_path)["input_ids"]
        print(len(dataset))

        logger.info("Constructing roBERTa style dataset")
        # merge multiple lines to form a single example
        merged_dataset = []
        
        # init the tmp with the first dataset
        tmp = dataset[0]

        for d in tqdm(dataset[1:]):
            # special case, empty line that indicates document breaks
            # i.e. [CLS] [SEP]
            # in this case, we want to keep the [SEP]
            if len(d) == 2:
                d.append(d[-1]) # convert to [CLS] [SEP] [SEP]
            
            d_len = len(d) - 2  # exclude the first [CLS] and last [SEP]

            if len(tmp) + d_len < self.max_len:
                # tmp = [CLS] xxx yyy zzz [SEP]
                # d = [CLS] aaa bbb [SEP]
                # resulting tmp = [CLS] xxx yyy zzz aaa bbb [SEP]

                # for a special case of d = [CLS] [SEP] [SEP]
                # resulting tmp will be:
                # [CLS] xxx yyy zzz [SEP] [SEP]
                # which later be added with the next sentence to form:
                # [CLS] xxx yyy zzz [SEP] ooo ppp [SEP]
                tmp = tmp[:-1] + d[1:]
            else:
                merged_dataset.append(tmp)
                tmp = d
        
        # add the leftover tmp
        merged_dataset.append(tmp)

        merged_dataset = [{"input_ids": d} for d in merged_dataset]
        
        return merged_dataset

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
