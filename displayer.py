from dataset import datasetRegister
from dataset import displayDatasetIterator
from dataset import displayGTAnnotations

images_path = "./dataset/normal/train"
annotations_filepath = "./dataset/normal/train/annotations.json"

datasetRegister("bees", images_path, annotations_filepath)

displayDatasetIterator("bees", False, 4, 2, None, [displayGTAnnotations])
