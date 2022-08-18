TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]

YAML_TEMPLATE = {
    "seed": 10,
    "output_dir": "",
    "tokenizer": {
        "pretrained": "",
    },
    "lm": {"model_type": ""},
    "glue": {
        "tasks": TASKS,
        "cola": {"pretrained": ""},
        "mrpc": {"pretrained": ""},
        "rte": {"pretrained": ""},
        "stsb": {"pretrained": ""},
        "sst2": {"pretrained": ""},
        "qnli": {"pretrained": ""},
        "mnli": {"pretrained": ""},
        "qqp": {"pretrained": ""},
    },
}

import os
import sys
import yaml

args = sys.argv

tokenizer_dir=args[1]
model_type=args[2]
best_model_dir = args[3]
output_dir = args[4]


# 1. Create Dir
print(">> 1. Create Output Dir", output_dir)
try:
    os.makedirs(output_dir)
except FileExistsError:
    print("Output dir", output_dir, "is not empty. Change before continue!")
    raise FileExistsError
YAML_TEMPLATE["output_dir"] = output_dir

# 2. Validate tokenizer_dir
print(">> 2. Validate tokenizer Dir", tokenizer_dir)
tokdirfiles = os.listdir(tokenizer_dir)
isexist = False
for f in tokdirfiles:
    if f == "vocab.txt":
        isexist = True
if not isexist:
    print("Tokenizer dir doesn't contains vocab.txt")
    raise FileNotFoundError
YAML_TEMPLATE["tokenizer"]["pretrained"] = tokenizer_dir


# 3. Check all models
print(">> 3. Use model_type", model_type, "- checking all models")
YAML_TEMPLATE["lm"]["model_type"] = model_type
import glob
for task in TASKS:
    ls = glob.glob(best_model_dir + "/" + task + "/checkpoint-*")
    ckpt = -1
    ckptdir = ""
    for l in ls:
        lckpt = l.split("-")[-1]
        ickpt = int(lckpt)
        if ckpt < 0 or ickpt < ckpt:
            ckpt = ickpt
            ckptdir = l
    YAML_TEMPLATE["glue"][task]["pretrained"] = ckptdir
    print(">>    ", task, "use", ckptdir)


with open("examples/gen_run.yaml", "w+") as fw:
    yaml.dump(YAML_TEMPLATE, fw)
