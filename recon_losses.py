import torch
import numpy as np

def concordance_cc(prediction, ground_truth):
    """Defines concordance loss for training the model.

    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """
    pred_mean = torch.mean(prediction)
    pred_var = torch.var(prediction, unbiased=False)

    gt_mean = torch.mean(ground_truth)
    gt_var = torch.var(ground_truth, unbiased=False)

    ccc = 0
    return 1 - ccc


