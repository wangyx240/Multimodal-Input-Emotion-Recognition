import torch
import numpy as np

def accuracy(prediction, ground_truth):
    """Defines concordance metric for model evaluation.

    Args:
       prediction: prediction of the model. size [
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """
    mean_pred = np.mean(prediction)
    mean_lab = np.mean(ground_truth)
    cov_pred = np.cov(prediction)
    cov_lab = np.cov(ground_truth)
    cov_pred_lab = np.cov(prediction, ground_truth)
    names_to_values, names_to_updates = map({
        'eval/mean_pred': mean_pred,
        'eval/mean_lab': mean_lab,
        'eval/cov_pred': cov_pred,
        'eval/cov_lab': cov_lab,
        'eval/cov_lab_pred': cov_pred_lab
    })
    denominator = (cov_pred + cov_lab + (mean_pred - mean_lab)**2)
    concordance_cc2 = (2 * cov_pred_lab) / denominator
    return concordance_cc2, names_to_values, names_to_updates
