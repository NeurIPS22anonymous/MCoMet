## Training

Run the following command to train a model using a specified config.
```bash
# Single GPU
conda activate mcomet
python main.py ${path-to-config}

# Multiple GPUs
conda activate mcomet
torchrun --nproc_per_node=${num-gpus} main.py ${path-to-config}
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.
```bash
conda activate mcomet
python main.py ${path-to-config} --checkpoint ${path-to-checkpoint} --eval
```


