## EFPN implementation based on detectron2

#### To train:

run training scripts such as frcnn_x101_efpn.py as included here, change configs accordingly


#### Modifications from detectron2:

1. "detectron2" folder name changed to "detectron" to correctly build (could also work outside of root dir)

2. in detectron/modeling/backbone, we modified resnet.py and fpn.py to create the EFPN and FTT module

3. slight modifications to fix import paths, cuda compatibility...etc, including in detectron/engine/defaults.py, detectron/layers/wrappers.py


#### Todo:
1. create loss function for supervised and independent training of EFPN+FTT 


#### Credits:

EFPN original paper: C. Deng, M. Wang, L. Liu, and Y. Liu.  Extended feature pyramid network for small objectdetection.CVPR, 2020.

(https://arxiv.org/pdf/2003.07021v1.pdf)


```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
