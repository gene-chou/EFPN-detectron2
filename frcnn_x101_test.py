#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2

from detectron.config import get_cfg # returns a copy of the default config
from detectron.engine import DefaultTrainer
from detectron.model_zoo import model_zoo


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# resnext ref: https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cc5d0adf648e
cfg.MODEL.RESNETS.NUM_GROUPS = 32 # 1 ==> ResNet; > 1 ==> ResNeXt
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 4
cfg.MODEL.RESNETS.DEPTH = 101
cfg.MODEL.DEVICE = 'cpu' 

#cfg.DATASETS.TRAIN = ("",)
#cfg.DATASETS.TEST = ("",) 

#cfg.SOLVER.IMS_PER_BATCH = 16 <-default
cfg.SOLVER.MAX_ITER = 1500 
cfg.SOLVER.STEPS = (1000, ) # The iteration number to decrease learning rate by GAMMA (0.1)


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()