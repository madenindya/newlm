import fire
import os
import logging
import torch
import random
import numpy as np

from newlm.utils.file_util import read_from_yaml
from newlm.lm.bert import TokenizerBuilder, LMBuilder
from newlm.glue.configs import GLUE_CONFIGS
from newlm.glue.cls_trainer import ClsTrainer


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
        self.__seed_all(self.config_dict.get("seed", 42))

    def __seed_all(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if "lm" in self.config_dict:
            self.config_dict["lm"]["hf_trainer"]["args"]["seed"] = seed
        if "glue" in self.config_dict:
            self.config_dict["glue"]["hf_trainer"]["args"]["seed"] = seed

    def run_pretrain(self):
        """
        Pre-trained BERT LM based on config file
        """

        logging.info("Build Tokenizer")
        tknz_builder = TokenizerBuilder(self.config_dict["tokenizer"]["config"])
        tknz_builder.create(
            input_dir=self.config_dict["tokenizer"]["input_dir"],
            output_dir=self.config_dict["tokenizer"]["output_dir"],
        )
        logging.info(
            "Save pre-trained tokenizer to", self.config_dict["tokenizer"]["output_dir"]
        )
        pretrain_tokenizer = self.config_dict["tokenizer"]["output_dir"]

        logging.info("Build LM using HuggingFace Trainer")
        lm_builder = LMBuilder(
            model_config=self.config_dict["lm"]["model"]["config"],
            tokenizer=pretrain_tokenizer,
            max_len=self.config_dict["lm"]["max_len"],
        )
        if "wandb" in self.config_dict:
            self.config_dict["lm"]["hf_trainer"]["args"]["run_name"] = (
                self.config_dict["wandb"].get("run_basename", "exp") + "-lm"
            )
        lm_builder.create(
            train_path=self.config_dict["lm"]["train_path"],
            output_dir=self.config_dict["lm"]["output_dir"],
            training_args=self.config_dict["lm"]["hf_trainer"]["args"],
        )
        logging.info("Save pre-trained LM to", self.config_dict["lm"]["output_dir"])
        pretrain_lm = self.config_dict["lm"]["output_dir"]

    def run_glue(self):
        """
        Run benchmark GLUE task based on config file
        """

        tasks = self.config_dict["glue"].get("tasks", GLUE_CONFIGS.keys())
        output_dir = self.config_dict["glue"]["output_dir"]
        training_args = self.config_dict["glue"]["hf_trainer"]["args"]

        cls_trainer = ClsTrainer(
            pretrained_model=self.config_dict["glue"]["pretrained_model"],
            pretrained_tokenizer=self.config_dict["glue"]["pretrained_tokenizer"],
            max_len=self.config_dict["glue"]["max_len"],
        )
        for task in tasks:
            if "wandb" in self.config_dict:
                self.config_dict["lm"]["hf_trainer"]["args"]["run_name"] = (
                    self.config_dict["wandb"].get("run_basename", "exp")
                    + f"-glue-{task}"
                )
            custom_args = training_args.copy()
            if task in self.config_dict["glue"]:
                custom_args.update(self.config_dict["glue"][task]["hf_trainer"]["args"])
            cls_trainer.train_and_eval(
                task=task,
                output_dir=f"{output_dir}/{task}/",
                training_args=custom_args,
            )

    # TODO: add script for run pretrain + downstream glue


if __name__ == "__main__":
    fire.Fire(ExperimentScript)
