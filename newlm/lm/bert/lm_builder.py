from typing import Union
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForPreTraining,
    BertTokenizerFast,
    PreTrainedTokenizer,
    LineByLineTextDataset,
    TextDatasetForNextSentencePrediction,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from ...utils.file_util import create_dir

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
                tokenizer, max_len=self.max_len
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
        use_nsp: bool = True,
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
            block_size=self.max_len,
        )

    def __get_dataset_nsp(self, train_path):
        return TextDatasetForNextSentencePrediction(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=self.max_len,
            nsp_probability=0.5,
        )
