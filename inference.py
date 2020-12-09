from detectron2.utils.visualizer import ColorMode, Visualizer
import os
import random
import cv2
from detectron.engine import DefaultPredictor
from detectron.modeling import build_model
from detectron2.config import get_cfg # returns a copy of the default config
#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.model_zoo import model_zoo
from detectron2.checkpoint import DetectionCheckpointer


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

#Use the final weights generated after successful training for inference  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

cfg.DATASETS.TEST = ("coco_2017_test")#datasets/coco/val2017")

cfg.MODEL.RESNETS.NUM_GROUPS = 32 # 1 ==> ResNet; > 1 ==> ResNeXt
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 8
cfg.MODEL.RESNETS.DEPTH = 101
cfg.MODEL.BACKBONE.FREEZE_AT = 1
cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
cfg.MODEL.DEVICE = 'cuda' 

cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5", "res6"]

cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.SOLVER.IMS_PER_BATCH = 1

cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 5000

cfg.INPUT.MIN_SIZE_TRAIN = (400,)
cfg.INPUT.MAX_SIZE_TRAIN = 667

cfg.MODEL.WEIGHTS = "output/model_final.pth"
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

#model = build_model(cfg)
#DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
#checkpointer = DetectionCheckpointer(model, save_dir = "inf_output")

predictor = DefaultPredictor(cfg)

#Call the COCO Evaluator function and pass the Validation Dataset
# evaluator = COCOEvaluator("boardetect_val", cfg, False, output_dir="/output/")
evaluator = COCOEvaluator("coco_2017_test", cfg, False, output_dir="inf_output/")
val_loader = build_detection_test_loader(cfg, "coco_2017_test", num_workers=4)

#Use the created predicted model in the previous step
print(inference_on_dataset(predictor.model, val_loader, evaluator))