Significant portions of the code were taken from the following github repo:
https://github.com/shakes76/GFNet

Uses gfnet-xs architecture

Since the original repo used the MIT License, a copy of the MIT License has also been included in this sub-folder.

Dependencies: pytorch, timm (requires searching on conda-forge)

Since we are using ADNI data which has two types, AD or NC, top 5 accuracy is not really useful.
TODO:
Remove Acc5
Train first verify later
Remove redundant progress checking prints
