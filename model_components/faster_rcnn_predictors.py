######
# Modified version of FastRCNNPredictor from: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
######

import torch
import torch.nn as nn
from pprint import pprint


class FasterRCNNPredictorNCDMaskOrig(nn.Module):

    def __init__(self, bbox_pred, prediction_model, cls_score_orig=None, cluster_mapping=None, l2_normalize=False,
                 num_classes=None, mask_type='orig_model', device='cuda'):
        super(FasterRCNNPredictorNCDMaskOrig, self).__init__()
        self.device = device
        self.bbox_pred = bbox_pred
        self.cls_score = prediction_model
        self.cluster_mapping = cluster_mapping  # num_classes (location is class_id, value is associated cluster_id)
        self.l2_normalize = l2_normalize
        self.cls_score_orig = cls_score_orig
        self.num_classes = num_classes
        self.mask_type = mask_type
        print('USING CLUSTER MAPPING ', cluster_mapping)
        pprint(cluster_mapping)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x_l2 = x.flatten(start_dim=1)
        if self.l2_normalize:
            x_l2 = torch.nn.functional.normalize(x_l2)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas = torch.cat([bbox_deltas for _ in range(self.num_classes)],
                                dim=1)  # duplicate for faster rcnn format

        # scores from original model used for background classifier
        scores2 = self.cls_score_orig(x)
        scores = self.cls_score.predict_scores(x_l2)

        # compute argmax over original model scores for background classifier
        scores_max0 = torch.where(torch.argmax(scores2, dim=1) == 0)[0]
        scores_nonmax0 = torch.where(torch.argmax(scores2, dim=1) != 0)[0]

        # compute minimum and maximum value over mini-batch for masking
        min_val = torch.min(scores)
        max_val = torch.max(scores)

        scores_new = torch.ones(scores.shape[0], scores.shape[1] + 1).to(self.device)
        if self.mask_type == 'orig_model':
            # use original model to determine if box is background
            scores_new[scores_max0, 0] *= max_val
            scores_new[scores_nonmax0, 0] *= min_val
            scores_new[:, 1:] = scores
        elif self.mask_type == 'none':
            # never pick background
            scores_new[:, 0] *= min_val
            scores_new[:, 1:] = scores
        else:
            raise NotImplementedError
        if self.cluster_mapping is not None:
            # use cluster mapping to reorganize network predictions
            # scores_new = [bg_score, cluster0_score, cluster1_score ...]
            s = scores_new[:, 1:]
            s = s[:, self.cluster_mapping]
            scores_new = torch.cat([scores_new[:, 0].unsqueeze(1), s], dim=1)

        return scores_new, bbox_deltas


class FasterRCNNPredictorOrig(nn.Module):

    def __init__(self, bbox_pred, cls_score, num_classes=None, device='cuda'):
        super(FasterRCNNPredictorOrig, self).__init__()
        self.device = device
        self.bbox_pred = bbox_pred
        self.cls_score = cls_score
        self.num_classes = num_classes

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas = torch.cat([bbox_deltas for _ in range(self.num_classes)],
                                dim=1)  # duplicate for faster rcnn format
        return scores, bbox_deltas


class FastRCNNPredictorClassAgnosticRegressor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)  # class agnostic
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas = torch.cat([bbox_deltas for _ in range(self.num_classes)], dim=1)

        return scores, bbox_deltas
