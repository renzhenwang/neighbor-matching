# neighbor-matching
This repository contains code for the paper "Neighbor Matching for Semi-supervised Learning", published at MICCAI 2021. 
The implementation is based on [LatentMixing](https://github.com/Prasanna1991/LatentMixing).


# How to run?
'''
python main_neighbor_matching.py --augu --out Final_models/ip1@350 --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 350 --lambda-u 1.0 --manualSeed 1 --noSharp --gpu 0 
'''
(For more detail, follow run.sh)

# Requirements:
1. PyTorch
2. pickle
3. PIL
4. torchvision
5. sklearn

(There might be more requirements but shouldn't be difficult to install them using conda.)

# Credit:
1. https://github.com/Prasanna1991/LatentMixing
2. https://github.com/YU1ut/MixMatch-pytorch

# Questions
Please feel free to contract "wrzhen@stu.xjtu.edu.cn".
