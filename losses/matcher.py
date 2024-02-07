

import torch
from scipy.optimize import linear_sum_assignment
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@torch.no_grad()
def hungarian_matcher(outputs, targets,
                      const_cost_bbox=1, const_cost_class=1, const_cost_giou=1):
    """
    Hungarian Matcher from
    https://github.com/facebookresearch/detr/blob/main/models/matcher.py

    Uses scipy.optimize.linear_sum_assignment() from
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html


    Parameters:
        outputs: This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        const_cost_bbox (int) = 1
        const_cost_class (int) = 1
        const_cost_giou (int) = 1


    Returns:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    """
    B, n_queries = outputs["pred_logits"].shape[:2]

    # We flatten to compute the cost matrices in a batch
    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B * n_queries, n_classes]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B * n_queries, 4]

    # Also concat the target labels and boxes
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])

    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -out_prob[:, tgt_ids]

    # Compute the L1 cost between boxes (coordinates)
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

    # Compute the giou cost betwen boxes (coordinates)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

    # Final cost matrix with shape [B, n_queries, -1]
    cost = const_cost_bbox * cost_bbox + const_cost_class * cost_class + const_cost_giou * cost_giou
    cost = cost.view(B, n_queries, -1).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

