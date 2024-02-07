

import torch.nn as nn


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """


