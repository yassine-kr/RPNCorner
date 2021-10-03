import os
import cv2
import matplotlib.pyplot as plt
import json
import random
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def getImageRecord(images_path, image_name, image_idx, image_data):
    record = {}
    filename = os.path.join(images_path, image_name)
    height, width = cv2.imread(filename).shape[:2]
    
    record["file_name"] = filename
    record["image_id"] = image_idx
    record["height"] = height
    record["width"] = width
    
    annos = image_data["annotations"]
    objs = []
    
    for anno in annos:
        obj = {
            "bbox": anno["bbox"],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    return record


def getImagesAnnotations(images_path, annotations_filepath):
    with open(annotations_filepath) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for (key, data) in imgs_anns.items():
        dataset_dicts.append(getImageRecord(images_path, key, key, data))
    return dataset_dicts

def datasetRegister(dataset_name, images_path, annotations_filepath):
    DatasetCatalog.register(dataset_name, lambda ip = images_path, af = annotations_filepath: getImagesAnnotations(ip, af))
    MetadataCatalog.get(dataset_name).set(thing_classes = ["bee"])


def displayGTAnnotations(visualizer, image, gt_annotations, processor_output):
    imageResult = None
    counter = 0
    for anno in gt_annotations:
        imageResult = visualizer.draw_box(anno["bbox"], edge_color=(1,1,1), alpha= 1)
        counter = counter + 1
    print("number of ground truth: ", counter)
    return imageResult

def displayProposals(visualizer, image, gt_annotations, processor_output):
    if processor_output == None:
        return None
    labels = [round(score.item(),2) for score in processor_output["proposals"].objectness_logits]
    return visualizer.overlay_instances(boxes=processor_output["proposals"].proposal_boxes.to("cpu"), labels = labels)

def displayPredictions(visualizer, image, gt_annotations, processor_output):
    if processor_output == None:
        return None
    return visualizer.draw_instance_predictions(processor_output["instances"].to("cpu"))
    

"""
iterate on images of already registred dataset

Args:
    dataset_name
	random(boolean):
		whether choose random images or follow order
	start (int):
        index of the first element to show
    count(int):
		number of images to process, -1 for all images
    processor:
        function to process, it should have the followinf signature (image)
    displayers:
        list of function to execute for displying data, each one shoud have the 
        following signature (visualizer, image, gt_annotations, processor_output)
        it should return the image_result and image
"""
def displayDatasetIterator(dataset_name, rand, start, count, processor, displayers):
    metadata = MetadataCatalog.get(dataset_name)
    dataset = DatasetCatalog.get(dataset_name)
    if count == -1:
        count = len(dataset)
    if start == -1:
        start = 0
    if rand == True:
        dicts = random.sample(dataset, count)
    else:
        dicts = DatasetCatalog.get(dataset_name)[start: start + count]
    for d in dicts:
        
        try:
            print(d["file_name"])
            image = cv2.imread(d["file_name"])
            visualizer = Visualizer(image, metadata=metadata)
            image_result = None
        
            output = None
            if processor != None:
                output = processor(image, d["image_id"])
        
            for displayer in displayers:
                image_result = displayer(visualizer, image, d["annotations"], output)

            if image_result != None:
                plt.figure()
                plt.imshow(image_result.get_image()[:,:,::-1])
                plt.show()
        except Exception as error:
            print(error)			    