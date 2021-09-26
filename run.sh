# -*- coding: utf-8 -*-

python main_neighbor_matching.py --augu --out Final_models/ip1@350 --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 350 --lambda-u 1.0 --manualSeed 1 --noSharp --gpu 0
python main_neighbor_matching_mixup.py --augu --out Final_models_mix/ip1@350 --epochs 256 --batch-size 128 --lr 0.0001 --schedule 50 125 --howManyLabelled 350 --lambda-u 1.0 --manualSeed 1 --noSharp --gpu 0 --alpha 1.0

