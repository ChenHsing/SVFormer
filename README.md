# SVFormer: Semi-supervised Video Transformer for Action Recognition

This is the official implementation of the paper [SVFormer](https://arxiv.org/abs/2211.13222)

```
@inproceedings{svformer,
  title={SVFormer: Semi-supervised Video Transformer for Action Recognition},
  author={Zhen Xing, Qi Dai, Han Hu, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang},
  booktitle={CVPR},
  year={2023}
}
```



## Installation

We tested the released code with the following conda environment

```
conda create -n svformer python=3.7
conda activate svformer
bash env.sh
```

## Data Preparation

We expect that `--train_list_path` and `--val_list_path` command line arguments to be a data list file of the following format
```
<path_1> <label_1>
<path_2> <label_2>
...
<path_n> <label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.
`--num_classes` should also be specified in the command line argument.

Additionally, `<path_i>` might be a relative path when `--data_root` is specified, and the actual path will be
relative to the path passed as `--data_root`.

We provide example as list_hmdb_40.



## Train script of SVFormer-B at Kinetic-400 1% setting

```
bash train.sh
```

## Main Results in paper 

This is an original-implementation for open-source use.
We are still re-running some models, and their scripts, checkpoints  will be released later.
In the following table we report the accuracy in original paper.

| Backbone   | UCF101-1% | UCF101-10% | Kinetic400-1% | Kinetic400-10% | 
| - | - |  - | - | - | 
| SVFormer-S | 31.4 | 79.1 | 32.6 | 61.6
| SVFormer-B | 46.3 | 86.7 | 49.1 | 69.4 


| Backbone   | HMDB51-40% | HMDB51-50% | HMDB51-60%|
| - | - |  - | - | 
| SVFormer-S | 56.2 | 58.2 | 59.7
| SVFormer-B | 61.6 | 64.4 | 68.2


## Acknowledgements

Our code is modified from [TimeSformer](https://github.com/facebookresearch/TimeSformer). Thanks for their awesome work!
