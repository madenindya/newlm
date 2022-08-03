import fire
import os
import torch
import random
import numpy as np
from pathlib import Path
from loguru import logger
from newlm.utils.file_util import read_from_yaml, is_dir_empty
from newlm.lm.bert import TokenizerBuilder, LMBuilder
from newlm.glue.configs import GLUE_CONFIGS, GlueConfig
from newlm.glue.cls_trainer import ClsTrainer
from newlm.lm.elmo.lm_builder import ELMOLMBuilder
from newlm.utils.file_util import create_dir

class ExperimentScript:
    def __init__(self, config_file: str):
        """
        Parameters
        ----------
        config_file : str
            path to yaml file
        """
        self.config_file = config_file
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

    def run_all(self):
        """
        Pre-traine BERT Tokenizer and LM
        Then, run downstream GLUE
        based on config file
        """
        model_out_dir = str(self.output_dir / "model")
        glue_out_dir = str(self.output_dir / "glue")

        self.__validate_train_lm(output_dir=model_out_dir)
        pretrain_tokenizer = self.__build_tokenizer(model_out_dir)
        pretrain_lm = self.__build_lm(pretrain_tokenizer, model_out_dir)

        self.config_dict["tokenizer"]["pretrained"] = pretrain_tokenizer
        self.config_dict["lm"]["pretrained"] = pretrain_lm
        self.run_glue()

    def run_pretrain(self):
        """
        Pre-trained BERT Tokenizer and LM based on config file
        """
        output_dir = str(self.output_dir / "model")
        self.__validate_train_lm(output_dir=output_dir)
        pretrain_tokenizer = self.__build_tokenizer(output_dir)
        self.__build_lm(pretrain_tokenizer, output_dir)

    # Do not use this function!
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
        self.__validate_train_lm(output_dir=output_dir)
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
        model_type = self.__get_model_type()

        if model_type == "bert":
            lm_builder = LMBuilder(
                model_config=self.config_dict["lm"]["model"]["config"],
                tokenizer=pretrain_tokenizer,
                max_len=self.config_dict["tokenizer"]["max_len"],
            )
        elif (
            model_type == "elmo-gpt"
            or model_type == "gpt2"
            or model_type == "bert-causal"
            or model_type == "bert-causal-r2l"
            or model_type == "elmo-bert-causal"
            or model_type == "elmo-bert-causal-v3"
            or model_type == "elmo-bert-causal-l2r-r2l"
            or model_type == "elmo-bert-causal-l2r-r2l-v2"
            or model_type == "elmo-bert-causal-l2r-r2l-v3"
            or model_type == "elmo-bert-causal-l2r-r2l-v4"
            or model_type == "elmo-bert-2-tower"
            or model_type == "elmo-bert-2-tower-v3"
        ):
            # We don't have to handle the exception (already handled from previous invocation)
            lm_builder = ELMOLMBuilder(
                model_config=self.config_dict["lm"]["model"]["config"],
                tokenizer=pretrain_tokenizer,
                max_len=self.config_dict["tokenizer"]["max_len"],
                model_type=model_type,
            )
        else:
            raise NotImplementedError(f"{model_type} is not implemented!")

        self.__rename_wandb("lm", self.config_dict["lm"]["hf_trainer"]["args"])
        self.__recalculate_batch_size(self.config_dict["lm"]["hf_trainer"])
        oth_args = self.config_dict["lm"]["model"].get("create_params", {})
        if oth_args is None:
            oth_args = {}

        lm_builder.create(
            train_path=self.config_dict["lm"]["train_path"],
            output_dir=output_dir,
            training_args=self.config_dict["lm"]["hf_trainer"]["args"],
            **oth_args,
        )
        logger.info(f"Save pre-trained LM to {output_dir}")
        pretrain_lm = output_dir
        return pretrain_lm

    def run_glue(self, seed=None, lr=None, bs=None, tasks=None):
        """
        Run benchmark GLUE task based on config file
        """

        if tasks is not None:
            self.config_dict["glue"]["tasks"] = tasks
        if bs is not None:
            logger.info(f"Replace total batch_size to {bs}")
            self.config_dict["glue"]["hf_trainer"]["total_batch_size"] = bs
            self.config_dict["glue"]["hf_trainer"]["args"]["per_device_eval_batch_size"] = bs
            eval_bs = self.__recalculate_eval_batch_size(bs)
            if eval_bs != bs:
                logger.info(f"Replace per device eval batch_size to {eval_bs}")
                self.config_dict["glue"]["hf_trainer"]["args"]["per_device_eval_batch_size"] = eval_bs
            self.output_dir = self.output_dir / f"bs_{bs}"
        if lr is not None:
            logger.info(f"Replace learning_rate to {lr}")
            self.config_dict["glue"]["hf_trainer"]["args"]["learning_rate"] = lr
            self.output_dir = self.output_dir / f"lr_{lr}"
        if seed is not None:
            logger.info(f"Replace seed to {seed}")
            self.config_dict["seed"] = seed
            self.__seed_all(seed)
            self.output_dir = self.output_dir / f"seed_{seed}"

        logger.info("Run Downstream GLUE")
        model_type = self.__get_model_type()

        tasks = self.config_dict["glue"].get("tasks", GLUE_CONFIGS.keys())
        output_dir = str(self.output_dir / "glue")
        batch_except = None
        try:
            self.__recalculate_batch_size(self.config_dict["glue"]["hf_trainer"])
        except Exception as e:
            batch_except = e
            logger.warning("Batch size is incorrect for default args. Make sure you define custom args!")
        hf_trainer_args = self.config_dict["glue"]["hf_trainer"]
        training_args = self.config_dict["glue"]["hf_trainer"]["args"]

        from_scratch = self.config_dict["glue"].get("from_scratch", False)
        pretrained_model, model_config = None, None
        lm_model = self.config_dict["lm"].get("model")
        model_config = lm_model.get("config", {}) if lm_model is not None else {}
        if not from_scratch:
            pretrained_model = self.__get_pt_lm_from_config()
        pretrained_tokenizer = self.__get_pt_tokenizer_from_config()

        cls_trainer = ClsTrainer(
            pretrained_model=pretrained_model,
            pretrained_tokenizer=pretrained_tokenizer,
            from_scratch=from_scratch,
            model_config=model_config,
            max_len=512,
            model_type=model_type,
        )
        for task in tasks:
            logger.info(f"Run GLUE {task}")
            custom_hf_args = hf_trainer_args.copy()
            custom_args = training_args.copy()
            self.__rename_wandb(f"glue-{task}", custom_args)
            oth_args = {}
            if task in self.config_dict["glue"]:
                if "hf_trainer" in self.config_dict["glue"][task]:
                    custom_hf_args.update(
                        self.config_dict["glue"][task]["hf_trainer"]
                    )
                    custom_args.update(
                        self.config_dict["glue"][task]["hf_trainer"]["args"]
                    )
                    custom_hf_args["args"] = custom_args
                    self.__recalculate_batch_size(custom_hf_args)
                    batch_except = None
                if "oth_args" in self.config_dict["glue"][task]:
                    oth_args = self.config_dict["glue"][task]["oth_args"]
            if batch_except is not None:
                raise batch_except
            cls_trainer.train_and_eval(
                task=task,
                output_dir=f"{output_dir}/{task}/",
                training_args=custom_hf_args["args"],
                oth_args=oth_args,
            )

    def run_glue_predict(self, test_data="validation", task=None):
        output_dir = self.output_dir / "glue-predict"
        model_type = self.__get_model_type()
        tasks = self.config_dict["glue"].get("tasks", GLUE_CONFIGS.keys())

        pretrained_tokenizer = self.__get_pt_tokenizer_from_config()

        def run_per_task(task):
            if "pretrained" not in self.config_dict["glue"][task]:
                logger.error(f"Please add glue.{task}.pretrained params in your config file")
                logger.warning(f"Skipping prediction for {task}")

            oth_args = {}
            if "oth_args" in self.config_dict["glue"][task]:
                oth_args = self.config_dict["glue"][task]["oth_args"]

            cls_trainer = ClsTrainer(
                pretrained_model=self.config_dict["glue"][task]["pretrained"],
                pretrained_tokenizer=pretrained_tokenizer,
                from_scratch=False,
                model_config=None,
                max_len=512,
                model_type=model_type,
            )
            task_output_dir = str(output_dir / task)
            cls_trainer.predict(task=task, output_dir=task_output_dir, oth_args=oth_args, test_data=test_data)

        if task is not None:
            run_per_task(task)
        else:
            for task in tasks:
                run_per_task(task)

        # For Test, run: python -m newlm run_glue_predict --config_file="examples/configs/run-predict-glue.yaml" --test_data="test"

    def run_predict_ensemble(self, test_data="validation"):
        ori_output_dir = self.output_dir

        # Run L2R
        self.output_dir = ori_output_dir / "l2r"
        self.config_dict["tokenizer"]["pretrained"] = self.config_dict["tokenizer"]["pretrained_l2r"]
        self.config_dict["lm"]["model_type"] = "bert-causal"
        for k in self.config_dict["glue"]:
            if "pretrained_l2r" in self.config_dict["glue"][k]:
                self.config_dict["glue"][k]["pretrained"] = self.config_dict["glue"][k]["pretrained_l2r"]
        self.run_glue_predict(test_data=test_data)

        # Run R2L
        self.output_dir = ori_output_dir / "r2l"
        self.config_dict["tokenizer"]["pretrained"] = self.config_dict["tokenizer"]["pretrained_r2l"]
        self.config_dict["lm"]["model_type"] = "bert-causal-r2l"
        for k in self.config_dict["glue"]:
            if "pretrained_r2l" in self.config_dict["glue"][k]:
                self.config_dict["glue"][k]["pretrained"] = self.config_dict["glue"][k]["pretrained_r2l"]
        self.run_glue_predict(test_data=test_data)

        # Run ensemble
        self.output_dir = ori_output_dir
        self.run_ensemble(base_dir=ori_output_dir, test_data=test_data)

    def run_ensemble(self, base_dir=None, l2r_r2l_ratio=[1,1], test_data="validation", merge_strategy=None):
        base_dir = str(self.output_dir) if base_dir is None else base_dir

        # merge ensemble
        tasks = self.config_dict["glue"].get("tasks", GLUE_CONFIGS.keys())
        for task in tasks:
            logger.info(f"Ensemble {task} with ratio {l2r_r2l_ratio}")
            if merge_strategy is not None and merge_strategy == "v2":
                self.merge_ensemble_v2(base_dir, task, l2r_r2l_ratio, test_data=test_data)
            else:
                self.merge_ensemble(base_dir, task, l2r_r2l_ratio, test_data=test_data)

    def merge_ensemble(self, base_dir, task, l2r_r2l_ratio=[1,1], test_data="validation"):
        import json
        import pandas as pd
        from datasets import load_metric

        sum_ratio = l2r_r2l_ratio[0] + l2r_r2l_ratio[1]
        l2r_weight = 1.0 * l2r_r2l_ratio[0] / sum_ratio
        r2l_weight = 1.0 * l2r_r2l_ratio[1] / sum_ratio

        glue_cfg = GlueConfig(task)

        output_dir = f"{str(self.output_dir)}/{l2r_weight}-{r2l_weight}"
        create_dir(output_dir)

        # Merge result
        l2r_path = f"{base_dir}/l2r/glue-predict/{task}/prob.csv"
        r2l_path = f"{base_dir}/r2l/glue-predict/{task}/prob.csv"
        merge_path = f"{output_dir}/ensemble_{task}.csv"
        ensemble_result_path = f"{output_dir}/ensemble_{task}_result.json"

        df_l2r = pd.read_csv(l2r_path, header=None)
        df_r2l = pd.read_csv(r2l_path, header=None)
        true_label_idx = glue_cfg.num_labels

        if (
            test_data == "validation" and
            (len(df_l2r[df_l2r[true_label_idx] != df_r2l[true_label_idx]]) > 0
            or len(df_r2l[df_l2r[true_label_idx] != df_r2l[true_label_idx]]) > 0)
        ):
            raise Exception("True label mismatch")

        df = pd.DataFrame()
        df["l2r_0"] = df_l2r[0]
        df["r2l_0"] = df_r2l[0]
        df["prob_0"] = (l2r_weight * df["l2r_0"]) + (r2l_weight * df["r2l_0"])
        if true_label_idx > 1:
            df["l2r_1"] = df_l2r[1]
            df["r2l_1"] = df_r2l[1]
            df["prob_1"] = (l2r_weight * df["l2r_1"]) + (r2l_weight * df["r2l_1"])
        if true_label_idx > 2:
            df["l2r_2"] = df_l2r[2]
            df["r2l_2"] = df_r2l[2]
            df["prob_2"] = (l2r_weight * df["l2r_2"]) + (r2l_weight * df["r2l_2"])
        if true_label_idx > 3:
            raise Exception("GLUE Not Handled")
        df["true_label"] = df_l2r[true_label_idx]

        keys = ["prob_0", "prob_1"]
        if "mnli" in task:
            keys.append("prob_2")
        if task != "stsb":
            pred_labels = []
            for i, row in df.iterrows():
                pred_label = np.argmax(row.get(keys).tolist())
                pred_labels.append(pred_label)
            df["pred_label"] = pred_labels
        else:
            df["pred_label"] = df["prob_0"]

        df.to_csv(merge_path, index=False)

        if test_data == "validation":
            ensemble_result = {}

            metric = load_metric("glue", task)
            ensemble_result["result"] = metric.compute(
                predictions=df["pred_label"], references=df["true_label"]
            )

            with open(ensemble_result_path, "w+") as fw:
                json.dump(ensemble_result, fw, indent=4)

    def run_predict_ensemble_v2(self, test_data="validation"):
        ori_output_dir = self.output_dir

        for task in self.config_dict["glue"]["tasks"]:
            print("RUN task", task)
            for model_type in self.config_dict["glue"][task]["ensembles"]:
                self.__replace_config_and_run(task, test_data, model_type, (ori_output_dir / model_type))

        # Run ensemble
        self.output_dir = ori_output_dir
        self.run_ensemble(base_dir=ori_output_dir, test_data=test_data, merge_strategy="v2")

    def __replace_config_and_run(self, task, test_data, model_type, output_dir):
        self.output_dir = output_dir
        self.config_dict["tokenizer"]["pretrained"] = self.config_dict["tokenizer"]["ensembles"][model_type]
        self.config_dict["lm"]["model_type"] = model_type

        pretraineds = self.config_dict["glue"][task]["ensembles"][model_type]
        for i, p in enumerate(pretraineds):
            self.output_dir = output_dir / str(i)
            self.config_dict["glue"][task]["pretrained"] = p
            self.run_glue_predict(task=task, test_data=test_data)


    def merge_ensemble_v2(self, base_dir, task, ratio=[1,1], test_data="validation"):
        import json
        import pandas as pd
        from datasets import load_metric

        sum_ratio = sum(ratio)
        data_weight = []
        for r in ratio:
            data_weight.append(1.0 * r / sum_ratio)

        glue_cfg = GlueConfig(task)

        output_dir = f"{str(self.output_dir)}/" + "-".join([str(w) for w in data_weight])
        create_dir(output_dir)

        # Merge result
        dfs = []
        i = 0
        for model_type in ["bert", "bert-causal", "bert-causal-r2l"]:
            i = 0
            while True:
                try:
                    dfs.append(pd.read_csv(f"{base_dir}/{model_type}/{str(i)}/glue-predict/{task}/prob.csv", header=None))
                    i += 1
                except:
                    break

        print(len(dfs), len(ratio))
        if len(dfs) != len(ratio):
            raise Exception("Ratio mismatch")

        merge_path = f"{output_dir}/ensemble_{task}.csv"
        ensemble_result_path = f"{output_dir}/ensemble_{task}_result.json"

        true_label_idx = glue_cfg.num_labels

        df = pd.DataFrame()
        prob = None
        for i, d in enumerate(dfs):
            df[f"{i}_0"] = d[0]
            prob = data_weight[i] * d[0] if i == 0 else (prob + data_weight[i] * d[0])
        df["prob_0"] = prob
        if true_label_idx > 1:
            for i, d in enumerate(dfs):
                df[f"{i}_1"] = d[1]
                prob = data_weight[i] * d[1] if i == 0 else (prob + data_weight[i] * d[1])
            df["prob_1"] = prob
        if true_label_idx > 2:
            for i, d in enumerate(dfs):
                df[f"{i}_2"] = d[2]
                prob = data_weight[i] * d[2] if i == 0 else (prob + data_weight[i] * d[2])
            df["prob_2"] = prob
        if true_label_idx > 3:
            raise Exception("GLUE Not Handled")
        df["true_label"] = dfs[0][true_label_idx]

        keys = ["prob_0", "prob_1"]
        if "mnli" in task:
            keys.append("prob_2")
        if task != "stsb":
            pred_labels = []
            for i, row in df.iterrows():
                pred_label = np.argmax(row.get(keys).tolist())
                pred_labels.append(pred_label)
            df["pred_label"] = pred_labels
        else:
            df["pred_label"] = df["prob_0"]

        df.to_csv(merge_path, index=False)

        if test_data == "validation":
            ensemble_result = {}

            metric = load_metric("glue", task)
            ensemble_result["result"] = metric.compute(
                predictions=df["pred_label"], references=df["true_label"]
            )

            with open(ensemble_result_path, "w+") as fw:
                json.dump(ensemble_result, fw, indent=4)

    def __get_model_type(self):
        model_type = self.config_dict["lm"].get("model_type", "bert")
        if model_type not in [
            "bert",
            "elmo-gpt",
            "gpt2",
            "bert-causal",
            "bert-causal-r2l",
            "elmo-bert-causal",
            "elmo-bert-causal-v3",
            "elmo-bert-causal-l2r-r2l",
            "elmo-bert-causal-l2r-r2l-v2",
            "elmo-bert-causal-l2r-r2l-v3",
            "elmo-bert-causal-l2r-r2l-v4",
            "elmo-bert-2-tower",
            "elmo-bert-2-tower-v3",
        ]:
            raise NotImplementedError(f"{model_type} is not implemented!")
        logger.info(f"Model type: {model_type}")
        return model_type

    def __recalculate_batch_size(self, hf_configs):
        """
        Recalculate based on: target batch, number of devices, per_device_train_batch_size
        Will update 'gradient_accumulation_steps'
        """
        if "total_batch_size" in hf_configs:
            total_batch_size = hf_configs["total_batch_size"]
            logger.info(f"Desired total batch: {total_batch_size}")

            num_device = 1
            if torch.cuda.is_available():
                num_device = torch.cuda.device_count()
            logger.info(f"Number of device: {num_device}")

            training_args = hf_configs["args"]

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

        return hf_configs

    def __recalculate_eval_batch_size(self, target):
        """
        Simply recalculate based on target batch and number of device
        """
        num_device = 1
        if torch.cuda.is_available():
            num_device = torch.cuda.device_count()
        if target % num_device > 0:
            raise Exception("Please recalculate your config batch (for eval)!")
        return int(target / num_device)

    def __get_pt_tokenizer_from_config(self):
        try:
            return self.config_dict["tokenizer"]["pretrained"]
        except:
            raise ValueError("Please add tokenizer.pretrained in your config file")

    def __get_pt_lm_from_config(self):
        try:
            return self.config_dict["lm"]["pretrained"]
        except:
            model_type = self.__get_model_type()
            if "elmo-bert-causal-l2r-r2l" in model_type:
                return (self.config_dict["lm"]["pretrained_l2r"], self.config_dict["lm"]["pretrained_r2l"])
            raise ValueError("Please add lm.pretrained in your config file")

    def __validate_train_lm(self, output_dir):
        is_outdir_empty = is_dir_empty(output_dir)
        is_resume = True
        try:
            create_params = self.config_dict["lm"]["model"]["create_params"]
            create_params["train_params"]["resume_from_checkpoint"]
        except:
            is_resume = False

        if not is_outdir_empty and not is_resume:
            raise Exception(
                f"Output directory '{output_dir}' is not empty! "
                + "To continue training LM, add config "
                + "'lm.model.create_params.train_params.resume_from_checkpoint'"
            )

    def __rename_wandb(self, task, training_args):
        runbasename = "exp"
        if "wandb" in self.config_dict:
            runbasename = self.config_dict["wandb"].get("run_basename", "exp")
        runfile = os.path.splitext(self.config_file)[0].split("/")[-1]
        runname = f"{runbasename}-{task}.{runfile}"
        training_args["run_name"] = runname


if __name__ == "__main__":
    fire.Fire(ExperimentScript)
