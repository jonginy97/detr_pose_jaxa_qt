# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP (Linear Sum Assignment Problem).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency, targets don't include "no_object" (background class). In general, there are more predictions 
    than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others remain unmatched 
    (and are thus treated as background).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, 
                 cost_position: float = 1, cost_orientation: float = 1):
        """
        Initializes the matcher.

        Params:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates in the matching cost.
            cost_giou: Relative weight of the GIoU loss of the bounding box in the matching cost.
            cost_position: Relative weight of the L1 error of the 3D position coordinates in the matching cost.
            cost_orientation: Relative weight of the quaternion orientation loss in the matching cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_position = cost_position
        self.cost_orientation = cost_orientation

        # Ensure that at least one cost is non-zero
        assert any([cost_class, cost_bbox, cost_giou, cost_position, cost_orientation]), \
            "All costs cannot be zero."

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.

        Params:
            outputs: Dictionary containing:
                - "pred_logits": Tensor of shape [batch_size, num_queries, num_classes] with the classification logits.
                - "pred_boxes": Tensor of shape [batch_size, num_queries, 4] with the predicted box coordinates.
                - "pred_positions": Tensor of shape [batch_size, num_queries, 3] with the predicted 3D positions.
                - "pred_orientations": Tensor of shape [batch_size, num_queries, 4] with the predicted quaternion orientations.

            targets: List of targets (len(targets) = batch_size), where each target is a dict containing:
                - "labels": Tensor of shape [num_target_boxes] with the class labels for each target object.
                - "boxes": Tensor of shape [num_target_boxes, 4] with the target box coordinates.
                - "position": Tensor of shape [num_target_boxes, 3] with the target 3D positions.
                - "orientation": Tensor of shape [num_target_boxes, 4] with the target quaternion orientations.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order).
                - index_j is the indices of the corresponding selected targets (in order).
            For each batch element, len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten the outputs to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # Shape: [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                # Shape: [batch_size * num_queries, 4]
        out_position = outputs["pred_positions"].flatten(0, 1)        # Shape: [batch_size * num_queries, 3]
        out_orientation = outputs["pred_orientations"].flatten(0, 1)  # Shape: [batch_size * num_queries, 4]

        # Concatenate target labels and boxes for all targets in the batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_position = torch.cat([v["positions"] for v in targets])
        tgt_orientation = torch.cat([v["orientations"] for v in targets])

        # Compute the classification cost. We use negative log probability of the target class.
        # The 1 is a constant that doesn't change the matching, so it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between bounding boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the GIoU cost between bounding boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the L1 cost between 3D positions
        cost_position = torch.cdist(out_position, tgt_position, p=1)

        # Compute the quaternion loss based on angle difference between orientations
        inner_product = torch.einsum('ij,kj->ik', out_orientation, tgt_orientation).clamp(-1, 1)
        angle_diff = 2 * torch.acos(torch.abs(inner_product))
        cost_orientation = angle_diff

        # Final cost matrix, weighted by the provided cost parameters
        C = (self.cost_bbox * cost_bbox 
             + self.cost_class * cost_class 
             + self.cost_giou * cost_giou 
             + self.cost_position * cost_position 
             + self.cost_orientation * cost_orientation)

        # Reshape the cost matrix for batch processing and apply Hungarian matching
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    """ Helper function to build the HungarianMatcher with the specified costs from args """
    return HungarianMatcher(
        cost_class=args.set_cost_class, 
        cost_bbox=args.set_cost_bbox, 
        cost_giou=args.set_cost_giou, 
        cost_position=args.set_cost_position,
        cost_orientation=args.set_cost_orientation
    )
