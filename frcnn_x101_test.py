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
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 8
cfg.MODEL.RESNETS.DEPTH = 101
cfg.MODEL.DEVICE = 'cpu' 

#dataset default is coco 
#cfg.DATASETS.TRAIN = ("",)
#cfg.DATASETS.TEST = ("",) 

cfg.INPUT.MIN_SIZE_TRAIN = (100,)
cfg.INPUT.MAX_SIZE_TRAIN = 166
cfg.INPUT.MIN_SIZE_TEST = (100,)
cfg.INPUT.MAX_SIZE_TEST = 166
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[2,4,8]]
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 16
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.SOLVER.IMS_PER_BATCH = 1

cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 1500 
cfg.SOLVER.STEPS = (1000, ) # The iteration number to decrease learning rate by GAMMA (0.1)


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()