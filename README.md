## NEWLM

Run train BERT LM:

```bash
python -m newlm run_pretrain --config_file="examples/configs/run.yaml"
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


Please check example file.
