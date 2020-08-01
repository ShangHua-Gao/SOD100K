# Cross Stage Fusion part in Highly Efficient Salient Object Detection with100K Parameters

## Introduction
The repo is the combination of Cross Stage Fusion (CSF) part of CSNet and Res2Net-50 backbone model.
Compared with existing ImageNet pre-trained large models, CSF achieves better performance with less parameters and FLOPS.


## Evaluation

You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).

## Usage

### Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)

### 1. Clone the repository

```shell
git clone https://github.com/MCG-NKU/SOD100K.git
cd SOD100K/CSF+Res2Net
```

### 2. Download the datasets

Download the following datasets and unzip them into `data` folder.

* [DUTS](https://drive.google.com/open?id=1immMDAPC9Eb2KCtGi6AdfvXvQJnSkHHo) dataset. The .lst file for training is `data/DUTS/DUTS-TR/train_pair.lst`.

### 3. Download the pre-trained models for backbone

Download the pretrained models of Res2Net50.
The pretrained model will be automatically downloaded. 
All you need to do is to set the default path of pytorch cache in `main.py`
```
res2net_path = '/home/shgao/.cache/torch/checkpoints/res2net50_v1b_26w_4s-3cf99910.pth'
```
[The pretrained model of res2net](https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net_v1b.py)
### 4. Train

1. Set the `--train_root` and `--train_list` path in `train.sh` correctly.

2. Train with a initial lr of 5e-5 for 24 epoches, which is divided by 10 after 15 epochs.
```shell
./train.sh
```
3. After training the result model will be stored under `results/run-*` folder.

### 5. Test

For single dataset testing: `*` changes accordingly and `--sal_mode` indicates different datasets (details can be found in `main.py`)
```shell
python main.py --mode='test' --model='results/run-*/models/final.pth' --test_fold='results/run-*-sal-e' --sal_mode='e'
```
For all datasets testing used in our paper: `0` indicates the gpu ID to use
```shell
./forward.sh 0 main.py results/run-*
```
All results saliency maps will be stored under `results/run-*-sal-*` folders in .png formats.


### 6. Pre-trained models and Prediction results

The pretrained models for CSF is now available on [ONEDRIVE](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/shgao_mail_nankai_edu_cn/EWYBOMYWNklLlVG38QlmozkBEIPgaCzsrrPq7BBFUEGjRg?e=ZTtbnM).

Note: Only support `bath_size=1`.

Prediction results on [ONEDRIVE](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/shgao_mail_nankai_edu_cn/EVBf1QE8Y5ZBhma4MEYABNMBnnMxOMiS6qAyPz3Kkv6bqA?e=jpcIjP).

## Citation
If you find this work or code is helpful in your research, please cite:
```
@InProceedings{gao2020sod100k,
  author = {Gao, Shang-Hua and Tan, Yong-Qiang and Cheng, Ming-Ming and Lu, Chengze and Chen, Yunpeng and Yan, Shuicheng},
  title = {Highly Efficient Salient Object Detection with 100K Parameters},
  booktitle = {ECCV},
  year = {2020},
}
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758}, 
}

```
