# Citation dataset

## Dependencies
- CUDA 10.0
- python 3.6.8
- pytorch 1.4.0
- torch-geometric 1.6.0

## Datasets
We use the [ogbn-proteins dataset](https://ogb.stanford.edu/docs/nodeprop/).
When you first run our script, the dataset will be downloaded automatically.

## Usage
To reproduce the results, run the following script.
```sh
python s_train.py
```

## Reference implementation
Codes are written based on [deeperGCN](https://github.com/lightaime/deep_gcns_torch)
