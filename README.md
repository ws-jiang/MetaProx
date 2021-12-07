# MetaProx

## Requirements
* python3
* pytorch 1.8
* yaml
* cvxpy, cvxpylayer
* numpy, pandas
* also: GPU


## Preparing data
* SINE: self-contained
* QMUL: refer to DKT github [DKT](https://github.com/BayesWatch/deep-kernel-transfer)
* Sale: download from [UCI](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)
* MiniImagenet: downlaod from [Optimization as a Model for Few-shot Learning](https://github.com/markdtw/meta-learning-lstm-pytorch)

after downloaded all the datasets, put them in the data directory.
The directories structure in the `data` should be
* QMUL/
    * images/
* sales/
    * Sales_Transactions_Dataset_Weekly.csv
* mini-imagenet/
    * images/
    * split/

## script
* run the program: refer to `run.sh` in the `code` directory.
* Results are presented in the `log` directory.

