import glob
import json
import sys

args = sys.argv

output_dir=args[1]
task = args[2]

print(f"Summarizing {task}'s runs from dir: {output_dir}")

all_summary = []

for filename in glob.glob(f'{output_dir}/*/*/*/glue/{task}/all_results.json'):
    file_tokens = filename.split("/")
    bs = int(file_tokens[-6].split("_")[1])
    lr = float(file_tokens[-5].split("_")[1])
    seed = int(file_tokens[-4].split("_")[1])
    with open(filename, "r+") as fr:
        result = json.loads(fr.read())
        result['bs'] = bs
        result['lr'] = lr
        result['seed'] = seed
        all_summary.append(result)

all_summary_str = json.dumps(all_summary, indent = 4)

# Writing to sample.json
with open(f"{output_dir}/all_summary_{task}.json", "w") as outfile:
    outfile.write(all_summary_str)
