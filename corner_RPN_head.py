from torch import nn
from detectron2.modeling.proposal_generator.rpn import RPN_HEAD_REGISTRY
from corner_anchor_generator import CornerAnchorGenerator
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.config import configurable
from typing import List
import torch.nn.functional as F
import torch

@RPN_HEAD_REGISTRY.register()
class Standard_RPNHead(nn.Module):
	@configurable
	def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
		super().__init__()
		
		self.convStandard= nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		
		self.objectness_logits_standard = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

		self.anchor_deltas_standard = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
		
		for l in [self.convStandard, self.objectness_logits_standard, self.anchor_deltas_standard]:
			nn.init.normal_(l.weight, std=0.01)
			nn.init.constant_(l.bias, 0)
		
			
	@classmethod
	def from_config(cls, cfg, input_shape):
		in_channels = [s.channels for s in input_shape]
		assert len(set(in_channels)) == 1, "Each level must have the same channel!"
		in_channels = in_channels[0]
		
		anchor_generator = build_anchor_generator(cfg, input_shape)
		num_anchors = anchor_generator.num_anchors
		box_dim = anchor_generator.box_dim
		assert (
			len(set(num_anchors)) == 1
		), "Each level must have the same number of anchors per spatial position"
		
		return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}
	
	def forward(self, features: List[torch.Tensor]):
		
		pred_objectness_logits_standard = []
		pred_anchor_deltas_standard = []
		for x in features:
			t_standard = F.relu(self.convStandard(x))
			pred_objectness_logits_standard.append(self.objectness_logits_standard(t_standard))
			pred_anchor_deltas_standard.append(self.anchor_deltas_standard(t_standard))
		return pred_objectness_logits_standard, pred_anchor_deltas_standard

@RPN_HEAD_REGISTRY.register()
class CornerDoubleRPNHead(nn.Module):
	@configurable
	def __init__(self, *, in_channels: int, num_center_anchors: int, num_corner_anchors: int, box_dim: int = 4):
		super().__init__()
		
		self.convCenter = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convCorner = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		
		self.objectness_logits_center = nn.Conv2d(in_channels, num_center_anchors, kernel_size=1, stride=1)
		self.objectness_logits_corner = nn.Conv2d(in_channels, num_corner_anchors, kernel_size=1, stride=1)

		self.anchor_deltas_center = nn.Conv2d(in_channels, num_center_anchors * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_corner = nn.Conv2d(in_channels, num_corner_anchors * box_dim, kernel_size=1, stride=1)
		
		for l in [self.convCenter, self.convCorner, self.objectness_logits_center, self.objectness_logits_corner, self.anchor_deltas_center, self.anchor_deltas_corner]:
			nn.init.normal_(l.weight, std=0.01)
			nn.init.constant_(l.bias, 0)
			
			
	@classmethod
	def from_config(cls, cfg, input_shape):
		in_channels = [s.channels for s in input_shape]
		assert len(set(in_channels)) == 1, "Each level must have the same channel!"
		in_channels = in_channels[0]
		
		center_anchor_generator = build_anchor_generator(cfg, input_shape)
		num_anchors_center = center_anchor_generator.num_anchors
		box_dim_center = center_anchor_generator.box_dim
		assert (
			len(set(num_anchors_center)) == 1
		), "Each level must have the same number of anchors per spatial position"
		
		corner_anchor_generator = CornerAnchorGenerator(cfg, input_shape)
		num_anchors_corner= corner_anchor_generator.num_anchors
		assert (
			len(set(num_anchors_corner)) == 1
		), "Each level must have the same number of anchors per spatial position"
		box_dim_corner = corner_anchor_generator.box_dim

		assert box_dim_center == box_dim_corner, "Box dims of center and corner anchors generator do not match"
		
		return {"in_channels": in_channels, "num_center_anchors": num_anchors_center[0], "num_corner_anchors": num_anchors_corner[0], "box_dim": box_dim_center}
	
	def forward(self, features: List[torch.Tensor]):
		
		pred_objectness_logits_center = []
		pred_anchor_deltas_center = []
		pred_objectness_logits_corner = []
		pred_anchor_deltas_corner = []
		for x in features:
			t_center = F.relu(self.convCenter(x))
			pred_objectness_logits_center.append(self.objectness_logits_center(t_center))
			pred_anchor_deltas_center.append(self.anchor_deltas_center(t_center))
			t_corner = F.relu(self.convCorner(x))
			pred_objectness_logits_corner.append(self.objectness_logits_corner(t_corner))
			pred_anchor_deltas_corner.append(self.anchor_deltas_corner(t_corner))
		return pred_objectness_logits_center, pred_anchor_deltas_center, pred_objectness_logits_corner, pred_anchor_deltas_corner



@RPN_HEAD_REGISTRY.register()
class CornerRPNHead(nn.Module):
	@configurable
	def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
		super().__init__()

		self.convCenter = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convTopLeft = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convTopRight = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convBottomLeft = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convBottomRight = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
	
		self.convs = [
			self.convCenter,
			self.convTopLeft,
			self.convTopRight,
			self.convBottomLeft,
			self.convBottomRight
		]

		self.objectness_logits_center = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_top_left = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_top_right = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_bottom_left = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_bottom_right = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

		self.objectness = [
			self.objectness_logits_center,
			self.objectness_logits_top_left,
			self.objectness_logits_top_right,
			self.objectness_logits_bottom_left,
			self.objectness_logits_bottom_right
		]

		self.anchor_deltas_center = nn.Conv2d(in_channels, num_anchors* box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_top_left = nn.Conv2d(in_channels, num_anchors  * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_top_right = nn.Conv2d(in_channels,num_anchors  * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_bottom_left = nn.Conv2d(in_channels, num_anchors  * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_bottom_right = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

		self.deltas = [
			self.anchor_deltas_center,
			self.anchor_deltas_top_left,
			self.anchor_deltas_top_right,
			self.anchor_deltas_bottom_left,
			self.anchor_deltas_bottom_right
		]

		for l in self.convs + self.objectness + self.deltas:
			nn.init.normal_(l.weight, std=0.01)
			nn.init.constant_(l.bias, 0)
			
			
	@classmethod
	def from_config(cls, cfg, input_shape):
		in_channels = [s.channels for s in input_shape]
		assert len(set(in_channels)) == 1, "Each level must have the same channel!"
		in_channels = in_channels[0]
		
		anchor_generator = build_anchor_generator(cfg, input_shape)
		num_anchors = anchor_generator.num_anchors
		
		assert (
			len(set(num_anchors)) == 1
		), "Each level must have the same number of anchors per spatial position"
		
		box_dim = anchor_generator.box_dim
		
		num_anchors = num_anchors[0]
		return {"in_channels": in_channels, "num_anchors": num_anchors, "box_dim": box_dim}
	
	def forward(self, features: List[torch.Tensor]):
		
		pred_objectness_logits = []
		pred_anchor_deltas = []
		for conv, objectness, deltas in zip(self.convs, self.objectness, self.deltas):
			logits_results = []
			deltas_results = []
			for x in features:
				t = F.relu(conv(x))
				logits_results.append(objectness(t)) 
				deltas_results.append(deltas(t))

			pred_objectness_logits.append(logits_results)
			pred_anchor_deltas.append(deltas_results)
			
		return pred_objectness_logits, pred_anchor_deltas


@RPN_HEAD_REGISTRY.register()
class CornerShiftRPNHead(nn.Module):
	@configurable
	def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
		super().__init__()
		
		self.convCenter = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convTopLeft = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convTopRight = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convBottomLeft = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
		self.convBottomRight = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
	
		self.convs = [
			self.convCenter,
			self.convTopLeft,
			self.convTopRight,
			self.convBottomLeft,
			self.convBottomRight
		]

		self.objectness_logits_center = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_top_left = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_top_right = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_bottom_left = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
		self.objectness_logits_bottom_right = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

		self.objectness = [
			self.objectness_logits_center,
			self.objectness_logits_top_left,
			self.objectness_logits_top_right,
			self.objectness_logits_bottom_left,
			self.objectness_logits_bottom_right
		]

		self.anchor_deltas_center = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_top_left = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_top_right = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_bottom_left = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
		self.anchor_deltas_bottom_right = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

		self.deltas = [
			self.anchor_deltas_center,
			self.anchor_deltas_top_left,
			self.anchor_deltas_top_right,
			self.anchor_deltas_bottom_left,
			self.anchor_deltas_bottom_right
		]

		for l in self.convs + self.objectness + self.deltas:
			nn.init.normal_(l.weight, std=0.01)
			nn.init.constant_(l.bias, 0)
			
			
	@classmethod
	def from_config(cls, cfg, input_shape):
		in_channels = [s.channels for s in input_shape]
		assert len(set(in_channels)) == 1, "Each level must have the same channel!"
		in_channels = in_channels[0]
		
		anchor_generator = build_anchor_generator(cfg, input_shape)
		num_anchors = anchor_generator.num_anchors
		
		assert (
			len(set(num_anchors)) == 1
		), "Each level must have the same number of anchors per spatial position"
		
		box_dim = anchor_generator.box_dim
		
		return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}
	
	def forward(self, features: List[torch.Tensor]):
		
		pred_objectness_logits = []
		pred_anchor_deltas = []
		for idx, (conv, objectness, deltas) in enumerate(zip(self.convs, self.objectness, self.deltas)):
			logits_results = []
			deltas_results = []
			for x in features:
				t = conv(x)
				if idx == 1:
					t = torch.roll(t, (-1, -1), (2, 3))
				elif idx == 2:
					t = torch.roll(t, (-1, 1), (2, 3))
				elif idx == 3:
					t = torch.roll(t, (1, -1), (2, 3))
				elif idx == 4:
					t = torch.roll(t, (1, 1), (2, 3))	
				t = F.relu(t)
				logits_results.append(objectness(t)) 
				deltas_results.append(deltas(t))

			pred_objectness_logits.append(logits_results)
			pred_anchor_deltas.append(deltas_results)
			
		return pred_objectness_logits, pred_anchor_deltas
