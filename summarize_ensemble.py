import glob
import json
import sys

args = sys.argv

output_dir=args[1]
task = args[2]

print(f"Summarizing {task}'s runs from dir: {output_dir}")

all_summary = []

for filename in glob.glob(f'{output_dir}/*/ensemble_{task}_result.json'):
    file_tokens = filename.split("/")
    l2r = float(file_tokens[-2].split("-")[0])
    r2l = float(file_tokens[-2].split("-")[1])
    # print(l2r, r2l)
    with open(filename, "r+") as fr:
        result = json.loads(fr.read())['result']
        result['l2r'] = l2r
        result['r2l'] = r2l
        all_summary.append(result)

all_summary_str = json.dumps(all_summary, indent = 4)

# Writing to sample.json
with open(f"{output_dir}/summary_ensemble_{task}.json", "w") as outfile:
    outfile.write(all_summary_str)
