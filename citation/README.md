# Citation dataset

## Dependencies
- CUDA 10.1
- python 3.6.8
- pytorch 1.7.0
- torch-geometric 1.6.1

## Datasets
The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed)
We use the same semi-supervised setting as [GCN](https://github.com/tkipf/gcn)

## Results
Testing accuracy summarized below.
| Dataset | Depth |  Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Cora       | 64 | 84.71  |
| Cite       | 64 | 72.83  |
| Pubmed       | 64 | 80.24  |

## Usage
To reproduce the results, run the following script
```sh
sh semi.sh
```

## Reference implementation
Codes are written based on [GCNII](https://github.com/chennnM/GCNII/blob/master)
