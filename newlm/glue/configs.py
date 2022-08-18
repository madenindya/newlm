GLUE_CONFIGS = {
    "cola": {
        "keys": ("sentence", None),
        "metric_name": "matthews_correlation",
        "fname": "CoLA"
    },
    "mnli": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "validation_key": "validation_matched",
        "test_key": "test_matched",
        "label_map": ["entailment", "neutral", "contradiction"],
        "fname": "MNLI-m"
    },
    "mnli-mm": {
        "keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "validation_key": "validation_mismatched",
        "test_key": "test_mismatched",
        "label_map": ["entailment", "neutral", "contradiction"],
        "fname": "MNLI-mm"
    },
    "mrpc": {"keys": ("sentence1", "sentence2"), "fname": "MPRC"}, # "label_map": ["not_equivalent", "equivalent"]},
    "qnli": {"keys": ("question", "sentence"), "label_map": ["entailment", "not_entailment"], "fname": "QNLI"},
    "qqp": {"keys": ("question1", "question2"), "fname": "QQP"}, # "label_map": ["not_duplicate", "duplicate"]},
    "rte": {"keys": ("sentence1", "sentence2"), "label_map": ["entailment", "not_entailment"], "fname": "RTE"},
    "sst2": {"keys": ("sentence", None), "fname": "SST-2"}, # "label_map": ["negative", "positive"]},
    "stsb": {
        "keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "metric_name": "pearson",
        "fname": "STS-B"
    },
    "wnli": {"keys": ("sentence1", "sentence2")} #, "label_map": ["not_entailment", "entailment"]},
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
        self.test_key = GLUE_CONFIGS.get(task).get("test_key", "test")
        self.label_map = GLUE_CONFIGS.get(task).get("label_map", None)
        self.fname = GLUE_CONFIGS.get(task).get("fname", task)
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
