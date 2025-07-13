import torch
import numpy as np
from typing import Dict
from .condition import ImagingCondition


class bSSFPImagingCondition(ImagingCondition):
    def condition_loss(self, condition_input: Dict[str, torch.Tensor],
                       pred_orig_sample: torch.Tensor):
        """
        Compute conditional loss for b-SSFP imaging.

        :param condition_input: Dictionary of conditions, in bSSFP, we use the condition_input['ssfp'] value only.
        :param pred_orig_sample: The estimated x0 in diffusion sample.
        :return: The conditional loss to guide the diffusion.
        """
        fa = condition_input["fa"]
        condition_input_scale = condition_input["input_scale"]
        condition = condition_input["ssfp"] * condition_input_scale
        condition = condition.to(pred_orig_sample)
        k, t1, t2 = self.rescale_prediction(pred_orig_sample)
        cos_fa = np.cos(fa)
        ssfp = k/(1+cos_fa+(1-cos_fa)*t1/torch.clamp(t2, 1e-4, None))
        loss = self.distance(condition, ssfp)
        return loss
