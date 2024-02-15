######
# Modified version of RoIHeads from: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
######

import torch
from torch import nn
from torch import Tensor
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torch.jit.annotations import Optional, List, Dict, Tuple
import torch.nn.functional as F
from collections import OrderedDict


class RoIHeadsModified(RoIHeads):
    def __init__(self,
                 box_roi_pool, box_head, box_predictor,
                 box_fg_iou_thresh, box_bg_iou_thresh,
                 box_batch_size_per_image, box_positive_fraction,
                 bbox_reg_weights,
                 box_score_thresh, box_nms_thresh, box_detections_per_img,
                 postprocess_type='standard', proba_norm='softmax'):
        self.postprocess_type = postprocess_type
        self.proba_norm = proba_norm
        super(RoIHeadsModified, self).__init__(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

    def postprocess_detections_modified(self,
                                        class_logits,  # type: Tensor
                                        box_regression,  # type: Tensor
                                        proposals,  # type: List[Tensor]
                                        image_shapes,  # type: List[Tuple[int, int]]
                                        objectnesses,
                                        ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression,
                                           proposals)  # boxes_per_image x num_classes + 1 (from somewhere else) x 4

        if self.proba_norm == 'softmax':
            pred_scores = F.softmax(class_logits, -1)
        elif self.proba_norm == 'l1':
            pred_scores = F.normalize(class_logits, p=1, dim=-1)
        else:
            raise NotImplementedError

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_probas = []
        all_labels = []
        all_objectnesses = []

        all_boxes, all_scores, all_probas, all_labels = self.postprocess_loop(pred_boxes_list, pred_scores_list,
                                                                              image_shapes, num_classes, device,
                                                                              all_boxes, all_scores, all_probas,
                                                                              all_labels)

        return all_boxes, all_scores, all_probas, all_labels, all_objectnesses

    def postprocess_loop(self, pred_boxes_list, pred_scores_list, image_shapes, num_classes, device,
                         all_boxes, all_scores, all_probas, all_labels):
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            num_boxes = len(boxes)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            box_id = torch.arange(num_boxes, device=device)
            box_id = box_id.unsqueeze(1).repeat(1, num_classes)

            probas = torch.clone(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            box_id = box_id[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            box_id = box_id.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, box_id = boxes[inds], scores[inds], labels[inds], box_id[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, box_id = boxes[keep], scores[keep], labels[keep], box_id[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, box_id = boxes[keep], scores[keep], labels[keep], box_id[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_probas.append(probas[box_id])  # grab probability vectors for the chosen boxes
            all_labels.append(labels)
        return all_boxes, all_scores, all_probas, all_labels

    def forward(self,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                objs=None,
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features.clone())

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            if self.postprocess_type == 'standard':
                boxes, scores, probas, labels, objs = self.postprocess_detections_modified(class_logits, box_regression,
                                                                                           proposals,
                                                                                           image_shapes, objs)
            else:
                raise NotImplementedError

            num_images = len(boxes)
            for i in range(num_images):
                results_dict = {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "probas": probas[i]
                }
                if objs != []:
                    results_dict["objs"] = objs[i]
                result.append(results_dict)

        return result, losses

    def get_features(self, features, proposals, image_shapes):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        return box_features, class_logits, box_regression


def get_box_features(model, images, targets, return_logits=False):
    with torch.no_grad():
        if targets is None:
            compute_proposals = True
        else:
            compute_proposals = False
        # grab features for image
        features_, proposals_, images_sizes_, targets_ = im2feat(model, images, targets,
                                                                 compute_proposals=compute_proposals)

        # get ground truth features
        if targets is None:
            proposals = proposals_
        else:
            proposals = [targets_[ii]['boxes'] for ii in range(len(targets_))]
        box_feats, logits, regress_coords = model.roi_heads.get_features(features_, proposals, images_sizes_)

        if return_logits:
            return box_feats, logits, regress_coords
        else:
            return box_feats


def im2feat(model, images, targets=None, compute_proposals=True):
    if model.training and targets is None:
        raise ValueError("In training mode, targets should be passed")
    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    if compute_proposals:
        proposals, proposal_losses = model.rpn(images, features, targets)
    else:
        proposals = None
    return features, proposals, images.image_sizes, targets
