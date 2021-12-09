import glob
import json
import sys

args = sys.argv

output_dir=args[1]
task = args[2]

print(f"Summarizing {task}'s runs from dir: {output_dir}")

all_summary = {}

for name in glob.glob(f'{output_dir}/*/*/*/glue/{task}/all_results.json'):
    filename = name
    file_tokens = filename.split("/")
    key = "-".join(file_tokens[-6:-3])
    with open(name, "r+") as fr:
        result = json.loads(fr.read())
        all_summary[key] = result

all_summary_str = json.dumps(all_summary, indent = 4)

# Writing to sample.json
with open(f"{output_dir}/all_summary_{task}.json", "w") as outfile:
    outfile.write(all_summary_str)
