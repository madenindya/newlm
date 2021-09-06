## NEWLM

Build Language Model and test it on GLUE dataset.

### Setup Local Env

Copy `.env.template` to `.env` and edit it if necessary. Then, run:

```bash
source .env
```

### PreTrained BERT

Run train BERT LM:

```bash
python -m newlm run_pretrain --config_file="examples/configs/run_lm_tokenizer.yaml"
```

<!--
To run the tokenizer and language model separately:

```bash
python -m newlm run_pretrain_tokenizer --config_file="examples/configs/run_tokenizer.yaml"
python -m newlm run_pretrain_model --config_file="examples/configs/run_lm.yaml"
```
-->

##### Batch Size

To run with desired total batch, simply add config `lm.hf_trainer.total_batch_size`.
Then, we would automatically calculate the `num_device * accum_step * batch_per_device` to match the `total_batch_size`.

### Run on GLUE

Test your model on GLUE dataset:

```bash
python -m newlm run_glue --config_file="examples/configs/run_glue.yaml"
```

### Run End-to-End

Build LM then run it on GLUE dataset:

```bash
python -m newlm run_all --config_file="examples/configs/run.yaml"
```

### Sample and train English LM

First, put all Wikipedia files and BookCorpus files into the following:

```
examples/data/wikipedia/
examples/data/books/
```

Then, sample the data (to 1 percent)

```bash
./shuffle-and-sample.sh 1
```

Then, train

```bash
python -m newlm run_pretrain --config_file="examples/configs/run.1-percent.yaml"
```

##### Important Notes

Please check and adjust all example files, before running your experiment.
