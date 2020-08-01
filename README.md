# SOD100K
The official repo of the ECCV 2020 paper [Highly Efficient Salient Object Detection with 100K Parameters](http://mftp.mmcheng.net/Papers/20EccvSal100k.pdf).

# CSNet
Salient object detection models often demand a considerable amount of computation cost to make precise prediction for each pixel, making them hardly applicable on low-power devices. In this paper, we aim to relieve the contradiction between computation cost and model performance by improving the network efficiency to a higher degree. We propose a flexible convolutional module, namely generalized OctConv (gOctConv), to efficiently utilize both in-stage and cross-stages multi-scale features, while reducing the representation redundancy by a novel dynamic weight decay scheme. The effective dynamic weight decay scheme stably boosts the sparsity of parameters during training, supports learnable number of channels for each scale in gOctConv, allowing 80% of parameters reduce with negligible performance drop. Utilizing gOctConv, we build an extremely light-weighted model, namely CSNet, which achieves comparable performance with ∼ 0.2% parameters (100k) of large models on popular salient object detection benchmarks.

# CSF + Res2Net
`Cross Stage Fusion (CSF) part + Res2Net` is in the `CSF+Res2Net` subfolder.
CSF can be integrated into ImageNet pre-trained large models such as Res2Net to achieve SOTA performance with less parameters.


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
