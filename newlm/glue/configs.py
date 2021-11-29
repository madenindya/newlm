GLUE_CONFIGS = {
    "cola": {
        "keys": ("sentence", None),
        "metric_name": "matthews_correlation",
    },
    "mnli": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "validation_key": "validation_matched",
    },
    "mnli-mm": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "validation_key": "validation_mismatched",
    },
    "mrpc": {"keys": ("sentence1", "sentence2")},
    "qnli": {"keys": ("question", "sentence")},
    "qqp": {"keys": ("question1", "question2")},
    "rte": {"keys": ("sentence1", "sentence2")},
    "sst2": {"keys": ("sentence", None)},
    "stsb": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "metric_name": "pearson",
    },
    "wnli": {"keys": ("sentence1", "sentence2")},
}


class GlueConfig:
    def __init__(self, task: str, oth_args={}):
        self.__validate_task(task)
        self.task = task
        self.actual_task = task if task != "mnli-mm" else "mnli"
        self.keys = GLUE_CONFIGS.get(task).get("keys")
        self.num_labels = GLUE_CONFIGS.get(task).get("num_labels", 2)
        self.metric_name = GLUE_CONFIGS.get(task).get("metric_name", "accuracy")
        self.training_key = "train"
        self.validation_key = GLUE_CONFIGS.get(task).get("validation_key", "validation")
        self.detokenizer = None
        if "detokenizer" in oth_args:
            if task != "mrpc":
                raise ValueError(f"Should not detokenize data for task {task}")
            if oth_args["detokenizer"] not in ["moses", "treebank"]:
                raise ValueError("Detokenizer available: moses, treebank")
            self.detokenizer = oth_args["detokenizer"]

    def __validate_task(self, task: str):
        if task not in GLUE_CONFIGS:
            raise ValueError(
                f"Task {task} not exist. Please check the available GLUE tasks"
            )
