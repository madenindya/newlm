import fire
import os
import logging

from newlm.utils.file_util import read_from_yaml
from newlm.lm.bert.tokenizer_builder import TokenizerBuilder
from newlm.lm.bert.lm_builder import LMBuilder


class ExperimentScript:
    def __init__(self, config_file: str):
        """
        Parameters
        ----------
        config_file : str
            path to yaml file
        """
        file_split_tup = os.path.splitext(config_file)
        file_ext = file_split_tup[-1]
        if file_ext == ".yaml" or file_ext == ".yml":
            self.config_dict = read_from_yaml(config_file)
        else:
            raise NotImplementedError(f"Extension {file_ext} is not supported")

    def run_pretrain(self):
        """
        Pre-trained BERT LM based on config file
        """

        logging.info("Build Tokenizer")
        tknz_builder = TokenizerBuilder(self.config_dict["tokenizer"]["config"])
        tknz_builder.create(
            self.config_dict["tokenizer"]["input_dir"],
            self.config_dict["tokenizer"]["output_dir"],
        )
        logging.info(
            "Save pre-trained tokenizer to", self.config_dict["tokenizer"]["output_dir"]
        )
        pretrain_tokenizer = self.config_dict["tokenizer"]["output_dir"]

        logging.info("Build LM using HuggingFace Trainer")
        lm_builder = LMBuilder(
            self.config_dict["lm"]["model"]["config"], pretrain_tokenizer
        )
        lm_builder.create(
            self.config_dict["lm"]["train_path"],
            self.config_dict["lm"]["output_dir"],
            self.config_dict["lm"]["hf_trainer"]["args"],
        )


if __name__ == "__main__":
    fire.Fire(ExperimentScript)
