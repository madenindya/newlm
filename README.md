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

Then, sample the data (to 5 percent)
```
./shuffle-and-sample.sh 5
```

Then, train
```
python -m newlm run_pretrain --config_file="examples/configs/run.5-percent.yaml"
```


Please check example file.
