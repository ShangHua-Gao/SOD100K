# CSNet in Highly Efficient Salient Object Detection with100K Parameters

## Introduction
The repo is the light-weight Salient Object Detection model CSNet.

## Evaluation

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

## Usage

### Prerequisites

- [Pytorch 1.0+](http://pytorch.org/)

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

### 5. Training Codes
Due to our contract with the company, the training code cannot be released.
If you want to follow our work for **Non profit academic research**,
you can email us to **state your purpose and ask for the training code**.
For commercial use, you can also email us to get more details.

```E-amil: shgao(at)live.com AND cmm(at)nankai.edu.cn```

If you do not receive my reply within 1 week, feel free to ask my again ( better with a different e-mail in case the mail was judged as spam by the system).

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
