# SOD100K
The official repo of the TPAMI 2021 paper [A Highly Efficient Model to Study the Semantics of Salient Object Detection](https://mftp.mmcheng.net/Papers/21PAMI-Sal100K.pdf) and ECCV 2020 paper [Highly Efficient Salient Object Detection with 100K Parameters](http://mftp.mmcheng.net/Papers/20EccvSal100k.pdf).

Visit our project page for more details and dicussion: https://mmcheng.net/sod100k/
# CSNet
CNN-based salient object detection (SOD) methods achieve impressive performance. However, the way semantic information is encoded in them and whether they are category-agnostic is less explored. One major obstacle in studying these questions is the fact that SOD models are built on top of the ImageNet pre-trained backbones which may cause information leakage and feature redundancy. To remedy this, here we first propose an extremely lightweight holistic model tied to the SOD task that can be freed from classification backbones and trained from scratch, and then employ it to study the semantics of SOD models. With the holistic network and representation redundancy reduction by a novel dynamic weight decay scheme, our model has only 100K parameters, ∼ 0.2% of parameters of large models, and performs on par with SOTA on popular SOD benchmarks. Using CSNet, we find that a) SOD and classification methods use different mechanisms, b) SOD models are category insensitive, c) ImageNet pre-training is not necessary for SOD training, and d) SOD models require far fewer parameters than the classification models. The source code is publicly available at https://mmcheng.net/sod100k/.



# CSF + Res2Net
`Cross Stage Fusion (CSF) part + Res2Net` is in the `CSF+Res2Net` subfolder.
CSF can be integrated into ImageNet pre-trained large models such as Res2Net to achieve SOTA performance with less parameters.


## Citation
If you find this work or code is helpful in your research, please cite:
```
@ARTICLE{21PAMI-Sal100K,
  author={Ming-Ming Cheng* and Shanghua Gao* and Ali Borji and Yong-Qiang Tan and Zheng Lin and Meng Wang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Highly Efficient Model to Study the Semantics of Salient Object Detection}, 
  year={2021},
  volume={},
  number={},
  pages={},
  doi={10.1109/TPAMI.2021.3107956}
}
@InProceedings{gao2020sod100k,
  author = {Gao, Shanghua and Tan, Yong-Qiang and Cheng, Ming-Ming and Lu, Chengze and Chen, Yunpeng and Yan, Shuicheng},
  title = {Highly Efficient Salient Object Detection with 100K Parameters},
  booktitle = {ECCV},
  year = {2020},
}
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shanghua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2021},
  doi={10.1109/TPAMI.2019.2938758}, 
}

```

## License

The source code is free for research and education use only. Any comercial use should get formal permission first.

