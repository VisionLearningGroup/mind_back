import inspect
import logging
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.layers import ShapeSpec, nonzero_tuple, cat
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.box_head import build_box_head#, FastRCNNConvFCHead #FastRCNNConvHEAD
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, StandardROIHeads,ROI_HEADS_REGISTRY

@ROI_HEADS_REGISTRY.register()
class ScoreOutputROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        sample_proposal = True,
        return_feats=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training and sample_proposal:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)

        del targets
        if self.training:
            if return_feats:
                losses, preds, feats = self._forward_box(features, proposals, return_feats=True)
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
                return proposals, losses, feats
            losses, preds = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses, preds
        else:
            predictions = self._forward_box(features, proposals)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], return_feats=False):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        if not return_feats:
            del box_features
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if return_feats:
                return losses, predictions, box_features
            else:
                return losses, predictions
        else:
            return predictions

