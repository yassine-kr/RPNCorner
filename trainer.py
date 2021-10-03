from dataset import datasetRegister
from rpn_evaluation import RPNEvaluator
from custom_evaluation import CustomCOCOEvaluator
from configuration import getTrainingConfiguration
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import DatasetEvaluators


class TrainerRPN(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name):
		return DatasetEvaluators([RPNEvaluator(dataset_name, cfg, False, output_dir="./output")])

class TrainerFaster_RCNN(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name):
		return DatasetEvaluators([CustomCOCOEvaluator(dataset_name, cfg, False, output_dir="./output")])


train_images_path = "./dataset/normal/train"
train_annotations_filepath = "./dataset/normal/train/annotations.json"

datasetRegister("bees_train", train_images_path, train_annotations_filepath)

test_images_path = "./dataset/normal/test"
test_annotations_filepath = "./dataset/normal/test/annotations.json"

datasetRegister("bees_test", test_images_path, test_annotations_filepath)

network_type = "Faster_RCNN"
config = getTrainingConfiguration(
network_type,
model_filepath = None,
training_dataset_name = "bees_train",
test_dataset_name = "bees_test",
number_iterations = 20,
rpn_type = "corner")

if network_type == "RPN":
	trainer = TrainerRPN(config)
elif network_type == "Faster_RCNN":
	trainer = TrainerFaster_RCNN(config)

trainer.resume_or_load(resume=False)
trainer.train()