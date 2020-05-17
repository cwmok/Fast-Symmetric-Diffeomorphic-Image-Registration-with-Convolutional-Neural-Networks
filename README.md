# Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks

This is offical Pytorch implementation of "Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks" (CVPR 2020), written by Tony C.W. Mok.

## Prerequisites
- `Python 3.5.2+`
- `Pytorch 1.3.0`
- `NumPy`
- `NiBabel`

This code has been tested with `Pytorch` and GTX1080TI GPU.

## Inference
```
python
```

## Training
If you want to train a new model using your own dataset, please define your own data generator for `train_SYMNet.py` and perform the following script.

```
python train_SYMNet.py
```

## Publication
If you find this repository useful, please cite:

- **Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks**
[Tony C.W. Mok](https://cwmok.github.io/ "Tony C.W. Mok"), Albert C.S. Chung
CVPR 2020. [eprint arXiv:2003.09514](https://arxiv.org/abs/2003.09514 "eprint arXiv:2003.09514")

## Acknowledgment
Some codes in this repository are modified from [IC-Net](https://github.com/zhangjun001/ICNet) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).

###### Keywords
Keywords: Diffeomorphic Image Registration, convolutional neural networks, alignment, Symmetric Image Registration
