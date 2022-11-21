# SVFormer: Semi-supervised Video Transformer for Action Recognition

This is the official implementation of the paper [SVFormer](https://arxiv.org/abs/2208.03550)

```
@article{svformer,
  title={SVFormer: Semi-supervised Video Transformer for Action Recognition},
  author={Zhen Xing, Qi Dai, Han Hu, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang},
  journal={arXiv preprint arXiv:},
  year={2022}
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

<!-- ## Backbone Preparation

CLIP weights need to be downloaded from [CLIP official repo](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L30)
and passed to the `--backbone_path` command line argument.

## Script Usage

Training and evaliation scripts are provided in the scripts folder.
Scripts should be ready to run once the environment is setup and 
`--backbone_path`, `--train_list_path` and `--val_list_path` are replaced with your own paths.

For other command line arguments please see the help message for usage.

## Kinetics-400 Main Results -->

<!-- This is a re-implementation for open-source use.
We are still re-running some models, and their scripts, weights and logs will be released later.
In the following table we report the re-run accuracy, which may be slightly different from the original paper (typically +/-0.1%)

| Backbone | Decoder Layers | #frames x stride | top-1 | top-5 | Script | Model | Log |
| - | - | - | - | - | - | - | - |
| ViT-B/16 | 4 | 8 x 16 | 82.8 | 95.8 | [script](scripts/train_k400_vitb16_8f_dec4x768.sh) | [google drive](https://drive.google.com/file/d/1DoGjvDdkJoSa9i-wq1lh6QoEZIa4xTB3/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1-9vgsXMpnWBI9MxQV7SSQhkPfLomoYY3/view?usp=sharing) |
| ViT-B/16 | 4 | 16 x 16 | 83.7 | 96.2 | [script](scripts/train_k400_vitb16_16f_dec4x768.sh) | [google drive](https://drive.google.com/file/d/1dax4qUIOEI_QzYXv31J-87cDkonQetVQ/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1l2ivY28jUpwSmafQZvwtUo7tvm42i0PL/view?usp=sharing) |
| ViT-B/16 | 4 | 32 x 8 | 84.3 | 96.6 | [script](scripts/train_k400_vitb16_32f_dec4x768.sh) | [google drive](https://drive.google.com/file/d/1fzFM5pD39Kfp8xRAJuWaXR9RALLmnoeU/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1X1ZOdSCxXVeMpNhr_bviNKlRfJa5SMD7/view?usp=sharing) |
| ViT-L/14 | 4 | 8 x 16 | 86.3 | 97.2 | [script](scripts/train_k400_vitl14_8f_dec4x1024.sh) | [google drive](https://drive.google.com/file/d/1AkdF4CkOVW2uiycCVqCxS397oYxNISAI/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1OJFBmaE_tAwTzG-4i0CLQmhwGnN0psx1/view?usp=sharing) |
| ViT-L/14 | 4 | 16 x 16 | 86.9 | 97.4 | [script](scripts/train_k400_vitl14_16f_dec4x1024.sh) | [google drive](https://drive.google.com/file/d/1CTV9geLD3HLWzByAQUOf_m0F_g2lE3rg/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1a2iC4tQvjWFMI3UrEv2chuHwVrF6p9YF/view?usp=sharing) |
| ViT-L/14 | 4 | 32 x 8 | 87.7 | 97.6 | [script](scripts/train_k400_vitl14_32f_dec4x1024.sh) | [google drive](https://drive.google.com/file/d/1zNFNCKwP5owakELlnTCD20cpVQBqgJrB/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1dK7qoz3McYrmfS09FfreXC-LjUM7l0u4/view?usp=sharing) |
| ViT-L/14 (336px) | 4 | 32 x 8 | 87.7 | 97.8 | | | | -->

<!-- ## Data Loading Speed

As the training process is fast, video frames are consumed at a very high rate.
For easier installation, the current version uses PyTorch-builtin data loaders.
They are not very efficient and can become a bottleneck when using ViT-B as backbones.
We provide a `--dummy_dataset` option to bypass actual video decoding for training speed measurement. 
The model accuracy should not be affected. 
Our internal data loader is pure C++-based and does not bottleneck training by much on a machine with 2x Xeon Gold 6148 CPUs and 4x V100 GPUs. -->


## Acknowledgements

Our code is modified from [TimeSformer](https://github.com/facebookresearch/TimeSformer). Thanks for their awesome work!
