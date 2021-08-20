## NEWLM

Build Language Model and test it on GLUE dataset.

### PreTrained BERT

Run train BERT LM:

```bash
python -m newlm run_pretrain --config_file="examples/configs/run_lm_tokenizer.yaml"
```

To run the tokenizer and language model separately:

```bash
python -m newlm run_pretrain_tokenizer --config_file="examples/configs/run_tokenizer.yaml"
python -m newlm run_pretrain_model --config_file="examples/configs/run_lm.yaml"
```

##### Batch Size

To run with desired total batch, simply add config `lm.hf_trainer.total_batch_size`.
Then, we would automatically calculate the `num_device * accum_step * batch_per_device` to match the `total_batch_size`.

### Run on GLUE

Test your model on GLUE dataset:

```bash
python -m newlm run_glue --config_file="examples/configs/run_glue.yaml"
```

### Sample and traing English LM

First, put all Wikipedia files and BookCorpus files into the following:

```
examples/data/wikipedia/
examples/data/books/
```

Then, sample the data (to 1 percent)

```
./shuffle-and-sample.sh 1
```

Then, train

```
python -m newlm run_pretrain --config_file="examples/configs/run.1-percent.yaml"
```

Notes: Please check all example files.

### Local Env

Copy `.env.template` to `.env` and edit it if necessary. Run:

```bash
source .env
```
