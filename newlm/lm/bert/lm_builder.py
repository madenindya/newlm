from typing import Union
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    PreTrainedTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from ...utils.file_util import create_dir


class LMBuilder:
    def __init__(self, model_config, tokenizer: Union[str, PreTrainedTokenizer]):
        self.model_config = model_config
        self.tokenizer = tokenizer
        if type(tokenizer) == str:
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer, max_len=512)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

    def create(self, train_path: str, output_dir: str, training_args: dict):
        """
        Train BERT MLM from scratch.
        Here, we utilize HuggingFace Trainer to train the model.

        Parameters
        ----------
        train_path : str
            Path to training file
        output_dir : str
            Path to output dir
        training_args : dict
            Training params based on transformers.TrainingArguments
        """
        dataset = self.__get_dataset(train_path)
        config = BertConfig(**self.model_config)
        model = BertForMaskedLM(config=config)

        create_dir(output_dir)
        args = TrainingArguments(output_dir=output_dir, **training_args)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)

    def __get_dataset(self, train_path):
        return LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=128,
        )
