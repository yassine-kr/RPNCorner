import math
from typing import List
import torch
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import BufferList
from detectron2.modeling.anchor_generator import ANCHOR_GENERATOR_REGISTRY


@ANCHOR_GENERATOR_REGISTRY.register()
class CornerAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)
		
	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for corner in [1,2,3,4]:
			for size in sizes:
				area = size ** 2.0
				for aspect_ratio in aspect_ratios:
					w = math.sqrt(area / aspect_ratio)
					h = aspect_ratio * w

					if corner == 1:
						x0, y0, x1, y1 = 0.0 - stride / 2.0, 0.0 - stride / 2.0, w / 1.0 - stride / 2.0, h / 1.0 - stride / 2.0		 
						anchors.append([x0, y0, x1, y1])
					elif corner == 2:
						x0, y0, x1, y1 = - w + stride / 2.0, - stride / 2.0 , stride / 2.0, h / 1.0 - stride / 2.0
						anchors.append([x0, y0, x1, y1])
					elif corner == 3:
						x0, y0, x1, y1 = - stride / 2.0, -h + stride / 2.0 , w / 1.0 - stride / 2.0, stride / 2.0
						anchors.append([x0, y0, x1, y1])
					elif corner == 4:
						x0, y0, x1, y1 = - w + stride / 2.0, -h + stride / 2.0 , stride / 2.0, stride / 2.0
						anchors.append([x0, y0, x1, y1])
					
		return torch.tensor(anchors)


class TopLeftAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)

	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for size in sizes:
			area = size ** 2.0
			for aspect_ratio in aspect_ratios:
				w = math.sqrt(area / aspect_ratio)
				h = aspect_ratio * w
				
				x0, y0, x1, y1 = - stride / 2.0, - stride / 2.0, w / 1.0 - stride / 2.0, h / 1.0 - stride / 2.0		 
				anchors.append([x0, y0, x1, y1])

		return torch.tensor(anchors)

class TopRightAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)

	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for size in sizes:
			area = size ** 2.0
			for aspect_ratio in aspect_ratios:
				w = math.sqrt(area / aspect_ratio)
				h = aspect_ratio * w
				
				x0, y0, x1, y1 = - w + stride / 2.0, - stride / 2.0 , stride / 2.0, h / 1.0 - stride / 2.0
				anchors.append([x0, y0, x1, y1])
				
		return torch.tensor(anchors)

class BottomLeftAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)

	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for size in sizes:
			area = size ** 2.0
			for aspect_ratio in aspect_ratios:
				w = math.sqrt(area / aspect_ratio)
				h = aspect_ratio * w
				
				x0, y0, x1, y1 = - stride / 2.0, - h / 1.0 + stride / 2.0 , w / 1.0 - stride / 2.0, stride / 2.0
				anchors.append([x0, y0, x1, y1])
				
		return torch.tensor(anchors)


class BottomRightAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)

	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for size in sizes:
			area = size ** 2.0
			for aspect_ratio in aspect_ratios:
				w = math.sqrt(area / aspect_ratio)
				h = aspect_ratio * w
				
				x0, y0, x1, y1 = - w / 1.0 + stride / 2.0, - h / 1.0 + stride / 2.0 , stride / 2.0, stride / 2.0
				anchors.append([x0, y0, x1, y1])
				
		return torch.tensor(anchors)

@ANCHOR_GENERATOR_REGISTRY.register()
class CornerCenterAnchorGenerator(DefaultAnchorGenerator):
	def __init__(self, cfg, input_shape: List[ShapeSpec]):
		super().__init__(cfg, input_shape)
		
	def _calculate_anchors(self, sizes, aspect_ratios):
		cell_anchors = [
			self.generate_cell_anchors(stride, s, a).float() for s, a, stride in zip(sizes, aspect_ratios, self.strides)
		]
		return BufferList(cell_anchors)
		
	def generate_cell_anchors(self, stride, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
		anchors = []
		for corner in [0,1,2,3,4]:
			for size in sizes:
				area = size ** 2.0
				for aspect_ratio in aspect_ratios:
					w = math.sqrt(area / aspect_ratio)
					h = aspect_ratio * w
					
					if corner == 0:
						x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
						anchors.append([x0, y0, x1, y1])
					elif corner == 1:
						x0, y0, x1, y1 = 0.0 - stride / 2.0, 0.0 - stride / 2.0, w / 1.0 - stride / 2.0, h / 1.0 - stride / 2.0		 
						anchors.append([x0, y0, x1, y1])
					elif corner == 2:
						x0, y0, x1, y1 = - w + stride / 2.0, - stride / 2.0 , stride / 2.0, h / 1.0 - stride / 2.0
						anchors.append([x0, y0, x1, y1])
					elif corner == 3:
						x0, y0, x1, y1 = - stride / 2.0, -h + stride / 2.0 , w / 1.0 - stride / 2.0, stride / 2.0
						anchors.append([x0, y0, x1, y1])
					elif corner == 4:
						x0, y0, x1, y1 = - w + stride / 2.0, -h + stride / 2.0 , stride / 2.0, stride / 2.0
						anchors.append([x0, y0, x1, y1])
					
		return torch.tensor(anchors)