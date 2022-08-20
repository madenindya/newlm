
# ---- Modify this ----

ensemble_dir="ensemble/512-l2r_2e-4-r2l_2e-4"

task_dir_map={
    "cola": "0.5-0.5",
    "mnli": "0.5-0.5",
    "mrpc": "0.5-0.5",
    "qnli": "0.5-0.5",
    "qqp": "0.5-0.5",
    "rte": "0.5-0.5",
    "sst2": "0.5-0.5",
    "stsb": "0.5-0.5",
}

# ---- End of modification ----

import pandas as pd
from newlm.glue.configs import GLUE_CONFIGS

def run(task, ratio):
    df = pd.read_csv(f"{ensemble_dir}/{ratio}/ensemble_{task}.csv")
    data_label = df['pred_label'] if "label_map" not in GLUE_CONFIGS[task] else [GLUE_CONFIGS[task]["label_map"][l] for l in df['pred_label']]
    f = open(f"{ensemble_dir}/{GLUE_CONFIGS[task]['fname']}.tsv", "w")
    f.write("index\tprediction\n")
    for i, row in df.iterrows():
        f.write(str(i) + "\t" + str(data_label[i]) + "\n")
    f.close()

for k in task_dir_map:
    print("Task", k, "ensemble with ratio", task_dir_map[k])
    run(k, task_dir_map[k])
