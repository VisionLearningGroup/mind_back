# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import GeneralizedRCNN

__all__ = ["CustomRCNN"]

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]],
                stop_grad: bool = False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.backbone(images.tensor, stop_grad=stop_grad)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses, _ = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def get_internal_feats(self, batched_inputs: List[Dict[str, torch.Tensor]],
                stop_grad: bool = False):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone.bottom_up.get_sigmoid(images.tensor)
        return features

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]],
                         key: str = "image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[key].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def compute_consist(self, batched_inputs:Tuple[Dict[str, torch.Tensor]], proposals=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs, "image")
        images_aug = self.preprocess_image(batched_inputs, "image_aug")
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features_aug = self.backbone(images_aug.tensor)
        if proposals is None:
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}
        else:
            proposal_losses = {}
        proposals, detector_losses, pred_clean = self.roi_heads(images, features, proposals, gt_instances)
        _, detector_losses_aug, pred_aug = self.roi_heads(images, features_aug, proposals, gt_instances, sample_proposal=False)
        losses = {}
        losses.update(detector_losses)
        detector_losses_aug = {k+"_aug":v for k,v in detector_losses_aug.items()}
        losses.update(detector_losses_aug)
        losses.update(proposal_losses)
        prob_clean = F.softmax(pred_clean[0], dim=0)
        prob_aug = F.softmax(pred_aug[0], dim=0)
        consitency = self.compute_consistency(prob_clean, prob_aug)
        consitency += self.compute_consistency(prob_aug, prob_clean)
        return losses, consitency

    def feature_reg(self, batched_inputs:Tuple[Dict[str, torch.Tensor]], model_ema, proposals=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        images_v1 = self.preprocess_image(batched_inputs, "image_aug_v1")
        images_v2 = self.preprocess_image(batched_inputs, "image_aug_v2")
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images_v1.tensor)
        with torch.no_grad():
            features_aug = model_ema.backbone(images_v2.tensor)
        if proposals is None:
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images_v1, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}
        else:
            proposal_losses = {}
        proposals, detector_losses, region_feats_v1 = self.roi_heads(images_v1, features, proposals, gt_instances,
                                                                  return_feats=True)
        _, _, region_feats_v2 = self.roi_heads(images_v1, features_aug, proposals, gt_instances,
                                               return_feats=True, sample_proposal=False)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        consitency = self.contrastive_loss(region_feats_v1, region_feats_v2)
        losses.update(consitency)
        return losses

    def compute_consistency(self, pred1, pred2):
        p_teacher = pred1.log()
        loss = F.kl_div(p_teacher, pred2, reduction='batchmean')
        return loss
    def contrastive_loss(self, feat1, feat2):
        feat1 = F.normalize(feat1)
        feat2 = F.normalize(feat2).detach()
        mat = torch.matmul(feat1, feat2.t()) / 0.07
        prob = F.softmax(mat, dim=0) + 1e-6
        loss = torch.mean(-torch.log(prob).diag())
        return {'loss_nce': loss}
