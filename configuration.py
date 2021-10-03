from detectron2.config import get_cfg
from detectron2 import model_zoo
from corner_RPN_head import Standard_RPNHead, CornerDoubleRPNHead, CornerRPNHead, CornerShiftRPNHead
from corner_rpn import CornerDoubleRPN, CornerRPN, CornerMergeRPN

def getTrainingConfiguration(network_type, 
	model_filepath, 
	training_dataset_name , 
	test_dataset_name, 
	number_iterations,
	rpn_type = None):

	#intialize configuration
	cfg = get_cfg()
	if network_type == "RPN":
		cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))	
	elif network_type == "Faster_RCNN": 
		cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))	

	#set model weights
	if model_filepath != None:
		cfg.MODEL.WEIGHTS = model_filepath
	else:
		if network_type == "RPN":
			cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/rpn_R_50_FPN_1x.yaml")
		elif network_type == "Faster_RCNN":
			cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
	
	#set train dataset
	cfg.DATASETS.TRAIN = (training_dataset_name, )

	#set test dataset
	cfg.DATASETS.TEST = (test_dataset_name, )
	cfg.TEST.EVAL_PERIOD = 100

	#set number of classes
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

	#set solver parameters
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.MAX_ITER = number_iterations
	
	#set proposal generator
	if rpn_type == "standard":
		cfg.MODEL.RPN.HEAD_NAME = "Standard_RPNHead"
	elif rpn_type == "corner_double":
		cfg.MODEL.RPN.HEAD_NAME = "CornerDoubleRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerDoubleRPN"
	elif rpn_type == "corner":
		cfg.MODEL.RPN.HEAD_NAME = "CornerRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerRPN"
	elif rpn_type == "corner_shift":
		cfg.MODEL.RPN.HEAD_NAME = "CornerShiftRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerRPN"
	elif rpn_type == "corner_merge":
		cfg.MODEL.RPN.HEAD_NAME = "CornerRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerMergeRPN"
	return cfg


def getInferenceConfiguration(network_type, 
	model_filepath,
	test_dataset_name,
	rpn_pre_nms = None,
	rpn_post_nms = None,
	rpn_nms_threshold = None,
	rcnn_nms_threshold = None,
	score_threshold = None,
	rpn_type = None,
	number_detections = None):

	#intialize configuration
	cfg = get_cfg()
	if network_type == "RPN":
		cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))	
	elif network_type == "Faster_RCNN": 
		cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))	
	
	
	#set model weights
	cfg.MODEL.WEIGHTS = model_filepath

	#set test dataset and image size
	cfg.DATASETS.TEST = (test_dataset_name, )
	cfg.INPUT.MIN_SIZE_TEST = 0

	#set number of classes
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


	#set NMS pre, post proposals and nms thresholds
	if rpn_pre_nms != None:
		cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = rpn_pre_nms
	if rpn_post_nms != None:
		cfg.MODEL.RPN.POST_NMS_TOPK_TEST = rpn_post_nms
	if rpn_nms_threshold != None:
		cfg.MODEL.RPN.NMS_THRESH = rpn_nms_threshold
	if rcnn_nms_threshold != None:
		cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = rcnn_nms_threshold

	#set ROI score threshold
	if score_threshold != None:
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

	if rpn_type == "standard":
		cfg.MODEL.RPN.HEAD_NAME = "Standard_RPNHead"
	elif rpn_type == "corner_double":
		cfg.MODEL.RPN.HEAD_NAME = "CornerDoubleRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerDoubleRPN"
	elif rpn_type == "corner":
		cfg.MODEL.RPN.HEAD_NAME = "CornerRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerRPN"
	elif rpn_type == "corner_shift":
		cfg.MODEL.RPN.HEAD_NAME = "CornerShiftRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerRPN"
	elif rpn_type == "corner_merge":
		cfg.MODEL.RPN.HEAD_NAME = "CornerRPNHead"
		cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CornerMergeRPN"
	
	if number_detections != None:
		cfg.TEST.DETECTIONS_PER_IMAGE = number_detections 
	
	return cfg

