import fire
import os
import torch
import random
import numpy as np
from pathlib import Path
from loguru import logger
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
        self.output_dir = Path(self.config_dict["output_dir"])

    def __seed_all(self, seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if "lm" in self.config_dict and "hf_trainer" in self.config_dict["lm"]:
            self.config_dict["lm"]["hf_trainer"]["args"]["seed"] = seed
        if "glue" in self.config_dict and "hf_trainer" in self.config_dict["glue"]:
            self.config_dict["glue"]["hf_trainer"]["args"]["seed"] = seed

    def run_pretrain(self):
        """
        Pre-trained BERT Tokenizer and LM based on config file
        """
        output_dir = str(self.output_dir / "model")
        pretrain_tokenizer = self.__build_tokenizer(output_dir)
        self.__build_lm(pretrain_tokenizer, output_dir)

    def run_pretrain_tokenizer(self):
        """
        Pre-trained BERT Tokenizer based on config file
        """
        output_dir = str(self.output_dir / "model")
        self.__build_tokenizer(output_dir)

    def run_pretrain_model(self):
        """
        Pre-trained BERT LM based on config file
        """
        output_dir = str(self.output_dir / "model")
        pretrain_tokenizer = self.__get_pt_tokenizer_from_config()
        self.__build_lm(pretrain_tokenizer, output_dir)

    def __build_tokenizer(self, output_dir: str) -> str:
        logger.info("Build Tokenizer")
        tknz_builder = TokenizerBuilder(self.config_dict["tokenizer"]["config"])
        tknz_builder.create(
            input_dir=self.config_dict["tokenizer"]["input_dir"],
            output_dir=output_dir,
        )
        logger.info(f"Save pre-trained tokenizer to {output_dir}")
        pretrain_tokenizer = output_dir
        return pretrain_tokenizer

    def __build_lm(self, pretrain_tokenizer: str, output_dir: str) -> str:
        logger.info("Build LM using HuggingFace Trainer")
        lm_builder = LMBuilder(
            model_config=self.config_dict["lm"]["model"]["config"],
            tokenizer=pretrain_tokenizer,
            max_len=self.config_dict["tokenizer"]["max_len"],
        )
        if "wandb" in self.config_dict:
            self.config_dict["lm"]["hf_trainer"]["args"]["run_name"] = (
                self.config_dict["wandb"].get("run_basename", "exp") + "-lm"
            )
        self.__recalculate_batch_size()
        lm_builder.create(
            train_path=self.config_dict["lm"]["train_path"],
            output_dir=output_dir,
            training_args=self.config_dict["lm"]["hf_trainer"]["args"],
        )
        logger.info(f"Save pre-trained LM to {output_dir}")
        pretrain_lm = output_dir
        return pretrain_lm

    def run_glue(self):
        """
        Run benchmark GLUE task based on config file
        """

        tasks = self.config_dict["glue"].get("tasks", GLUE_CONFIGS.keys())
        output_dir = str(self.output_dir / "glue")
        training_args = self.config_dict["glue"]["hf_trainer"]["args"]

        pretrained_model = self.__get_pt_lm_from_config()
        pretrained_tokenizer = self.__get_pt_tokenizer_from_config()

        cls_trainer = ClsTrainer(
            pretrained_model=pretrained_model,
            pretrained_tokenizer=pretrained_tokenizer,
            max_len=self.config_dict["tokenizer"]["max_len"],
        )
        for task in tasks:
            logger.info(f"Run GLUE {task}")
            custom_args = training_args.copy()
            if "wandb" in self.config_dict:
                custom_args["run_name"] = (
                    self.config_dict["wandb"].get("run_basename", "exp")
                    + f"-glue-{task}"
                )
            if task in self.config_dict["glue"]:
                custom_args.update(self.config_dict["glue"][task]["hf_trainer"]["args"])
            cls_trainer.train_and_eval(
                task=task,
                output_dir=f"{output_dir}/{task}/",
                training_args=custom_args,
            )

    def __recalculate_batch_size(self):
        if "total_batch_size" in self.config_dict["lm"]["hf_trainer"]:
            total_batch_size = self.config_dict["lm"]["hf_trainer"]["total_batch_size"]
            logger.info(f"Desired total batch: {total_batch_size}")

            num_device = 1
            if torch.cuda.is_available():
                num_device = torch.cuda.device_count()
            logger.info(f"Number of device: {num_device}")

            training_args = self.config_dict["lm"]["hf_trainer"]["args"]

            if "per_device_train_batch_size" not in training_args:
                logger.warning(
                    "lm.hf_trainer.args.per_device_train_batch_size is not specified."
                    + " Automatically set to default 16."
                )
            batch_per_device = training_args.get("per_device_train_batch_size", 16)
            logger.info(f"Number of train batch per device: {batch_per_device}")

            total_device_batch = num_device * batch_per_device
            if total_batch_size % total_device_batch > 0:
                raise Exception("Please recalculate your config batch!")
            grad_accum_steps = int(total_batch_size / total_device_batch)
            logger.info(f"Set gradient_accumulation_steps to: {grad_accum_steps}")

            training_args["per_device_train_batch_size"] = batch_per_device
            training_args["gradient_accumulation_steps"] = grad_accum_steps

    def __get_pt_tokenizer_from_config(self):
        try:
            return self.config_dict["tokenizer"]["pretrained"]
        except:
            raise ValueError("Please add tokenizer.pretrained in your config file")

    def __get_pt_lm_from_config(self):
        try:
            return self.config_dict.get["lm"]["pretrained"]
        except:
            raise ValueError("Please add lm.pretrained in your config file")

    # TODO: add script for run pretrain + downstream glue


if __name__ == "__main__":
    fire.Fire(ExperimentScript)
