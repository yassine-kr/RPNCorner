from corner_anchor_generator import CornerAnchorGenerator, TopLeftAnchorGenerator, TopRightAnchorGenerator, BottomLeftAnchorGenerator, BottomRightAnchorGenerator, CornerCenterAnchorGenerator
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from typing import Dict, List, Optional
from detectron2.structures import ImageList, Instances
import torch
from detectron2.layers import ShapeSpec
from detectron2.layers import cat
from detectron2.structures import Boxes, Instances


@PROPOSAL_GENERATOR_REGISTRY.register()
class CornerDoubleRPN(RPN):

	def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
		super().__init__(cfg, input_shape)
		self.input_shape = input_shape
		self.cfg = cfg
		self.corner_generator = CornerAnchorGenerator(cfg, list(input_shape.values()))

	def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: Optional[List[Instances]] = None,):
		
		features = [features[f] for f in self.in_features]
		
		center_anchors = self.anchor_generator(features)
		corner_anchors = self.corner_generator(features)
		
		pred_objectness_logits_center, pred_anchor_deltas_center, pred_objectness_logits_corner, pred_anchor_deltas_corner = self.rpn_head(features)
		
		# Transpose the Hi*Wi*A dimension to the middle:
		pred_objectness_logits_center = [
			# (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
			score.permute(0, 2, 3, 1).flatten(1)
			for score in pred_objectness_logits_center
		]
		
		pred_anchor_deltas_center = [
			# (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
			x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
			.permute(0, 3, 4, 1, 2)
			.flatten(1, -2)
			for x in pred_anchor_deltas_center
		]

		# Transpose the Hi*Wi*A dimension to the middle:
		pred_objectness_logits_corner = [
			# (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
			score.permute(0, 2, 3, 1).flatten(1)
			for score in pred_objectness_logits_corner
		]
		
		pred_anchor_deltas_corner = [
			# (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
			x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
			.permute(0, 3, 4, 1, 2)
			.flatten(1, -2)
			for x in pred_anchor_deltas_corner
		]

		if self.training:
			assert gt_instances is not None, "RPN requires gt_instances in training!"
			gt_labels_center, gt_boxes_center = self.label_and_sample_anchors(center_anchors, gt_instances)
			losses_center = self.losses(
				center_anchors, pred_objectness_logits_center, gt_labels_center, pred_anchor_deltas_center, gt_boxes_center
			)
			gt_labels_corner, gt_boxes_corner = self.label_and_sample_anchors(corner_anchors, gt_instances)
			losses_corner = self.losses(
				corner_anchors, pred_objectness_logits_corner, gt_labels_corner, pred_anchor_deltas_corner, gt_boxes_corner
			)
			losses = {}
			losses["loss_rpn_cls_center"] = losses_center["loss_rpn_cls"]
			losses["loss_rpn_loc_center"] = losses_center["loss_rpn_loc"]
			losses["loss_rpn_cls_corner"] = losses_corner["loss_rpn_cls"]
			losses["loss_rpn_loc_corner"] = losses_corner["loss_rpn_loc"]
		else:
			losses = {}

		
		proposals_center = self.predict_proposals(
			center_anchors, pred_objectness_logits_center, pred_anchor_deltas_center, images.image_sizes
		)
		proposals_corner = self.predict_proposals(
			corner_anchors, pred_objectness_logits_corner, pred_anchor_deltas_corner, images.image_sizes
		)
		
		proposals = []
		for proposals_ci,proposals_bi in zip(proposals_center, proposals_corner):
			proposals.append(Instances.cat([proposals_ci,proposals_bi ]))
		
		return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class CornerRPN(RPN):

	def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
		super().__init__(cfg, input_shape)
		self.input_shape = input_shape
		self.cfg = cfg
	
		self.top_left_generator = TopLeftAnchorGenerator(cfg, list(input_shape.values()))
		self.top_right_generator = TopRightAnchorGenerator(cfg, list(input_shape.values()))
		self.bottom_left_generator = BottomLeftAnchorGenerator(cfg, list(input_shape.values()))
		self.bottom_right_generator = BottomRightAnchorGenerator(cfg, list(input_shape.values()))

	def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: Optional[List[Instances]] = None,):
		
		features = [features[f] for f in self.in_features]
		
		anchors = [
			self.anchor_generator(features),
			self.top_left_generator(features),
			self.top_right_generator(features),
			self.bottom_left_generator(features),
			self.bottom_right_generator(features)
		]
		
		pred_objectness_logits, pred_anchor_deltas, = self.rpn_head(features)
		
		# Transpose the Hi*Wi*A dimension to the middle:
		for index in range(len(pred_objectness_logits)):
			pred_objectness_logits[index] = [
				# (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
				score.permute(0, 2, 3, 1).flatten(1)
				for score in pred_objectness_logits[index]
			]
		
		for index in range(len(pred_anchor_deltas)):
			pred_anchor_deltas[index] = [
				# (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
				x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
				.permute(0, 3, 4, 1, 2)
				.flatten(1, -2)
				for x in pred_anchor_deltas[index]
			]

		if self.training:
			assert gt_instances is not None, "RPN requires gt_instances in training!"
	
			losses = {}
			names_cls = [
				"loss_rpn_cls_center",
				"loss_rpn_cls_top_left",
				"loss_rpn_cls_top_right",
				"loss_rpn_cls_bottom_left",
				"loss_rpn_cls_bottom_right",
			]
			names_loc = [
				"loss_rpn_loc_center",
				"loss_rpn_loc_top_left",
				"loss_rpn_loc_top_right",
				"loss_rpn_loc_bottom_left",
				"loss_rpn_loc_bottom_right",
			]
			for anc, pred_logits, pred_deltas, name_cls, name_loc in zip(anchors, pred_objectness_logits, pred_anchor_deltas, names_cls, names_loc):
				gt_labels, gt_boxes = self.label_and_sample_anchors(anc, gt_instances)
		
				loss = self.losses(
					anc, pred_logits, gt_labels, pred_deltas, gt_boxes
				)

				losses[name_cls] = loss["loss_rpn_cls"]
				losses[name_loc] = loss["loss_rpn_loc"]
		else:
			losses = {}

		positions_proposals = []
		for anc, pred_logits, pred_deltas in zip(anchors, pred_objectness_logits, pred_anchor_deltas):
			positions_proposals.append(self.predict_proposals(
				anc, pred_logits, pred_deltas, images.image_sizes
			))
		
		proposals = []
		for i in range(len(images)):
			proposals.append(Instances.cat([p[i] for p in positions_proposals]))
		
		return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class CornerMergeRPN(RPN):

	def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
		super().__init__(cfg, input_shape)
		self.input_shape = input_shape
		self.cfg = cfg
		self.corner_generator = CornerCenterAnchorGenerator(cfg, list(input_shape.values()))

	
	def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: Optional[List[Instances]] = None,):

		features = [features[f] for f in self.in_features]
		
		anchors = self.corner_generator(features)

		pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
		
		
		results_logits = []
		for i in range(len(pred_objectness_logits[0])):
			results_logits.append(torch.cat([pred[i] for pred in pred_objectness_logits], 1))

		pred_objectness_logits = results_logits		

		results_deltas = []
		for i in range(len(pred_anchor_deltas[0])):
			results_deltas.append(torch.cat([pred[i] for pred in pred_anchor_deltas], 1))
		
		pred_anchor_deltas = results_deltas

		# Transpose the Hi*Wi*A dimension to the middle:
		pred_objectness_logits = [
			# (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
			score.permute(0, 2, 3, 1).flatten(1)
			for score in pred_objectness_logits
		]
		
		pred_anchor_deltas = [
			# (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
			x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
			.permute(0, 3, 4, 1, 2)
			.flatten(1, -2)
			for x in pred_anchor_deltas
		]
		
		
		if self.training:
			assert gt_instances is not None, "RPN requires gt_instances in training!"
			
			gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
			
			losses = self.losses(
				anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
			)
		else:
			losses = {}

		
		proposals = self.predict_proposals(
			anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
		)
		return proposals, losses