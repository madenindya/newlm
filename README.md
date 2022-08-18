## NEWLM

Build Language Model and test it on GLUE dataset.

### Setup

We recommend creating a new `conda` environment before running this exepriment.

```bash
pip install -r requirements.txt
wandb login
source .env
```

- Copy `.env.template` to `.env` and edit it if necessary.
- We use `wandb` to track our running experiment. This program is still runnable without it.
- At the time we run this experiement, we use `python=3.8`

### Run Experiments

We provide several script options to perform LM Pretraining. All script would need to receive config file in `.yaml` format. Please check the example file accordingly.

##### Pretrain LM

Run pretrain LM from scratch given a dataset.

```bash
python -m newlm run_pretrain --config_file="examples/configs_gcloud/run-100-percent.bert-causal.yaml"
```

##### Pretrain LM with given Tokenizer

Run pretrain LM with prebuild tokenizer.

```bash
python -m newlm run_pretrain_model --config_file="examples/configs_gcloud/run-100-percent.bert-causal-r2l.yaml"
```

##### Finetune Downstream GLUE

Run finetuning on GLUE dataset.

```bash
python -m newlm run_glue --config_file="examples/configs_gcloud/run-ft.bert.yaml"
```

##### Run All (Pretrain and Downstream)

Run pretrain LM, follow by finetuning with GLUE dataset.

```bash
python -m newlm run_all --config_file="examples/configs/run.yaml"
```

##### Run Prediction and Save Precomputed Proba

```bash
python -m newlm run_glue_predict --config_file="examples/configs/run-predict-ensemble.yaml"
```

##### Run Prediction (L2R:R2L) and Ensemble (1:1)

```bash
python -m newlm run_predict_ensemble --config_file="examples/configs/run-predict-ensemble.yaml"
```

##### Run Ensemble from Saved Precomputed Proba

```bash
python -m newlm run_ensemble --config_file="examples/configs/run-predict-ensemble.yaml" --l2r_r2l_ratio=[1,1]]
```

#### Config .yaml

We use huggingface's transformers library as our base library, so most of the config would follow it. But, we do have some additional config to ease our training process.

Here are some further details and explanation for the config file.

**Batch Size (Training)**

```
lm.hf_trainer.total_batch_size
```

To run with desired total batch. We would automatically calculate the `num_device * accum_step * batch_per_device` to match the `total_batch_size`

**Resume Training**

```
lm.model.create_params.train_params.resume_from_checkpoint: latest
```

We prevent a pretrain model to be saved in a non-empty directory. To resume training from the latest checkpoint, set the following config.

#### Grid Search

We also provide several script for performing grid search. Please adjust accordingly.

**Grid search finetune GLUE**

```bash
./run_grid-bert-all.sh
```

**Grid search ratio of BERT (L2R:R2L)**

```bash
./run_grid_ensemble.sh
```

### Run Ensemble

TBD

### Run Predict

TBD

### Run Predict Ensemble

TBD

## Available Scripts

### ./run_predict_glue_test.sh

- Run prediction on test dataset
- Prepare and create file to be submit to gluebenchmark

**Steps**

1. Prepare the dir that contains vocab.txt file.
   - Can use the one from pretrained model
   - or use one best_model's checkpoint
2. Prepare your best model & Put it under 1 directory. Expected structure:

```
best_model_dir
    ` cola
        `checkpoint-123
    ` mrpc
        `checkpoint-456
        `checkpoint-789
    etc
```

3. Open script and modify necessary fields
4. Run script and wait until finish
5. It would generate `submission.zip` under output_dir/glue-predict
6. Submit the file to gluebenchmark web
