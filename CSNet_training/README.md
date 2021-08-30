# CSNet training code
## Introduction
The repo is the light-weight Salient Object Detection model CSNet.

Salient object detection models often demand a considerable amount of computation cost to make precise prediction for each pixel, making them hardly applicable on low-power devices. In this paper, we aim to relieve the contradiction between computation cost and model performance by improving the network efficiency to a higher degree. We propose a flexible convolutional module, namely generalized OctConv (gOctConv), to efficiently utilize both in-stage and cross-stages multi-scale features, while reducing the representation redundancy by a novel dynamic weight decay scheme. The effective dynamic weight decay scheme stably boosts the sparsity of parameters during training, supports learnable number of channels for each scale in gOctConv, allowing 80% of parameters reduce with negligible performance drop. Utilizing gOctConv, we build an extremely light-weighted model, namely CSNet, which **achieves comparable performance with ∼ 0.2% parameters (100k) of large models** on popular salient object detection benchmarks.

## Evaluation

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

## Usage

### Prerequisites

- [Pytorch 1.0+](http://pytorch.org/) (We use PyTorch 1.0 for training, other version are not tested.)

### 1. Clone the repository

```shell
git clone https://github.com/MCG-NKU/SOD100K.git
cd SOD100K/CSNet
```

### 2. Download the datasets

Download the following datasets and unzip them into `dataset` folder.
Note that for each dataset, put images and GT into the folder 'images' and 'GT'.
Example:
```
SOD100K/CSNet/datasets/sal/DUTS-TE/GT
SOD100K/CSNet/datasets/sal/DUTS-TE/images
```

### 3. Testing

```
python test.py --config configs/csnet-L-x2.yml
```

### 4. Pre-trained models
Since the CSNet is extremely small, we just include the pre-trained model of CSNet in the `checkpoints` folder.

### 5. Training
**Non Profit Academic Research ONLY**

For commercial use, you can also email us to get more details.
```
python train.py --config configs/csnet-L-x2_train.yml

python finetune.py --config configs/csnet-L-x2_train.yml --epoch (the epoch you want to finetune)
```

```E-amil: shgao(at)live.com AND cmm(at)nankai.edu.cn```

## Citation
If you find this work or code is helpful in your research, please cite:
```
@ARTICLE{21PAMI-Sal100K,
  author={Ming-Ming Cheng and Shang-Hua Gao and Ali Borji and Yong-Qiang Tan and Zheng Lin and Meng Wang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Highly Efficient Model to Study the Semantics of Salient Object Detection}, 
  year={2021},
  volume={},
  number={},
  pages={},
  doi={10.1109/TPAMI.2021.3107956}
}
@InProceedings{gao2020sod100k,
  author = {Gao, Shang-Hua and Tan, Yong-Qiang and Cheng, Ming-Ming and Lu, Chengze and Chen, Yunpeng and Yan, Shuicheng},
  title = {Highly Efficient Salient Object Detection with 100K Parameters},
  booktitle = {ECCV},
  year = {2020},
}

```
