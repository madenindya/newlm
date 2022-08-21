
TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]

YAML_TEMPLATE = {
    "seed": 10,
    "output_dir": "",
    "tokenizer": {
        "pretrained": "",
        "ensembles": {}
    },
    "lm": {"model_type": ""},
    "glue": {
        "tasks": TASKS,
    }
}

import yaml
import sys

args = sys.argv
output_dir=args[1]

YAML_TEMPLATE["output_dir"] = output_dir

with open("examples/gen_run_ens.yaml", "w+") as fw:
    yaml.dump(YAML_TEMPLATE, fw)
