from dataset import datasetRegister
from configuration import getInferenceConfiguration
from rpn_evaluation import RPNEvaluator
from custom_evaluation import CustomCOCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import DatasetEvaluators
from detectron2.modeling import build_model


images_path = "./dataset/normal/test"
annotations_filepath = "./dataset/normal/test/annotations.json"

datasetRegister("bees", images_path, annotations_filepath)

network_type = "Faster_RCNN"
config = getInferenceConfiguration(network_type,
"./output/model_final.pth",
"bees",
rpn_nms_threshold = None,
rpn_pre_nms = None,
rpn_post_nms = None,
score_threshold = None,
rpn_type = "corner_merge",
number_detections = None)

if network_type == "RPN" or network_type == "RPN_C4":
	evaluator = RPNEvaluator("bees", config, False, output_dir="./output")
else:
	evaluator = CustomCOCOEvaluator("bees", config, False, output_dir="./output")


val_loader = build_detection_test_loader(config, "bees")
model = build_model(config)
DetectionCheckpointer(model).load(config.MODEL.WEIGHTS)


print(inference_on_dataset(model, val_loader, DatasetEvaluators([evaluator])))