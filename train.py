import os
import torch
import cv2

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import random

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


register_coco_instances("panorama", {}, "../../training_set/amsterdamai.json", "../../training_set/labels")
register_coco_instances("panoramatest", {}, "../../training_set/amsterdamaitest.json", "../../training_set/labels")
panorama_metadata = MetadataCatalog.get("panorama")

print(torch.cuda.is_available())

dataset_dicts = DatasetCatalog.get("panorama")

cfg = get_cfg()
cfg.merge_from_file(
    "../configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"
)
cfg.DATASETS.TRAIN = ("panorama",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("panoramatest", )
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.SOLVER.MAX_ITER = (
 144000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 3 classes (data, fig, hazelnut)

predictor = DefaultPredictor(cfg)
os.makedirs("testOutput", exist_ok=True)
evaluator = COCOEvaluator("panoramatest", cfg, False, output_dir='testOutput')
val_loader = build_detection_test_loader(cfg, "panoramatest")
inference_on_dataset(predictor.model, val_loader, evaluator)


# DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
# checkpointer = DetectionCheckpointer(model, save_dir="output")

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# torch.save(trainer.build_model(cfg).state_dict(), "weights.pth")
# taken from https://github.com/facebookresearch/detectron2/issues/35

# for d in random.sample(dataset_dicts, 3):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=panorama_metadata,
#                    scale=0.8,
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("name", v.get_image()[:, :, ::-1])
#     cv2.waitKey()
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=panorama_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("name",vis.get_image()[:, :, ::-1])
#     cv2.waitKey()
