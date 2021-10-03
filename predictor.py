from dataset import datasetRegister
from configuration import getInferenceConfiguration
from dataset import displayDatasetIterator
from dataset import displayGTAnnotations
from dataset import displayProposals
from dataset import displayPredictions

import time
from detectron2.engine import DefaultPredictor

images_path = "./dataset/normal/test"
annotations_filepath = "./dataset/normal/test/annotations.json"

datasetRegister("bees", images_path, annotations_filepath)


network_type = "Faster_RCNN"
config = getInferenceConfiguration(network_type,
"./output/model_final.pth",
"bees",
rcnn_nms_threshold = 0.3,
rpn_nms_threshold = None,
rpn_pre_nms = None,
rpn_post_nms = None,
score_threshold = None,
rpn_type = "corner_merge",
number_detections = None)


predictor = DefaultPredictor(config)
def processor(image, imageName):
	start_time = time.time()
	output = predictor(image)
	print("--- %s total time ---" % (time.time() - start_time))
	if network_type == "Faster_RCNN" or network_type == "Faster_RCNN_C4":
		print("number of predicted instances: ", len(output["instances"]))
	else:
		print("number of predicted proposals: ", len(output["proposals"]))
	return output

displayDatasetIterator("bees", False, 1, 1, processor, [displayGTAnnotations,displayPredictions ])
