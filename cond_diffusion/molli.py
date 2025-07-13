import torch
import numpy as np
from typing import Dict, Optional, Union, Sequence
from mri.bloch import look_locker
from .condition import ImagingCondition, get_likelihood


class LookLockerImagingCondition(ImagingCondition):
    def __init__(self, rho_rescale: float = 0.6,
                 t1_rescale: float = 2800.,
                 t2_rescale: float = 350.,
                 sigma: float = 0.12,
                 likelihood: str = "gaussian",
                 baseline_likelihood: str = 'gaussian',
                 baseline_sigma: str = 0.05,
                 t1_likelihood: str = 'laplacian',
                 t1_sigma: float = 100.,
                 z_init: Optional[torch.Tensor] = None,
                 z_init_reg: float = 0.2):
        """
        Reverse imaging for the T1 mapping sequence MOLLI, complementing MOLLI with the PD and T2 estimation. The fitting of MOLLI readouts must have been performed before conditioning. MOLLI fits a 3-param model :math:`A(1-INV\cdot\exp(-TI/T1^*)`. :math:`T1` and `A` will be used to guide the reverse diffusion process.

        :param rho_rescale: Rescaler for diffusion model outputs, rho = z[0] / rho_rescale.
        :param t1_rescale: T1 = z[1] * t1_rescale
        :param t2_rescale: T2 = z[2] * t2_rescale
        :param sigma: Not used.
        :param likelihood: Not used.
        :param baseline_likelihood: Type of the likelihood for the bSSFP image A in molli fitting.
        :param baseline_sigma: Likelihood STD.
        :param t1_likelihood: The likelihood of the estimated T1 by checking its deviation from the fitted T1 in MOLLI.
        :param t1_sigma: Tolerance of T1 deviation.
        :param z_init: Initial guess for z.
        :param z_init_reg: Penalizes the deviation of z estimates from the initial guess.
        """
        super().__init__(rho_rescale, t1_rescale, t2_rescale, sigma, likelihood)
        self.t1_sigma = t1_sigma
        self.t1_distance = get_likelihood(t1_sigma, t1_likelihood)
        self.baseline_sigma = baseline_sigma
        self.baseline_distance = get_likelihood(self.baseline_sigma, baseline_likelihood)
        self.z_init = z_init
        self.z_init_reg = z_init_reg

    def condition_loss(self, condition_input: Dict[str, torch.Tensor],
                       pred_orig_sample: torch.Tensor):
        """
        Compute conditional loss for look-locker imaging.

        :param condition_input: Dictionary of conditions.
        :param pred_orig_sample: The estimated z0.
        :return: The conditional loss to guide the diffusion.
        """
        fa = condition_input["fa"]
        condition_input_scale = condition_input["input_scale"]

        condition = condition_input["ssfp"] * condition_input_scale
        condition = condition.to(pred_orig_sample)
        init_t1_guess = condition_input["t1"]  # in seconds
        k, t1, t2 = self.rescale_prediction(pred_orig_sample)

        # bSSFP conditioner
        cos_fa = np.cos(fa)
        ssfp = k / (1 + cos_fa + (1 - cos_fa) * t1 / torch.clamp(t2, 1e-4, None))
        ssfp_loss = self.distance(condition, ssfp)

        # don't ignore the T1 from MOLLI fitting, it is more reliable!
        t1_loss = self.t1_distance(init_t1_guess, t1 * 1e-3)
        loss = t1_loss + ssfp_loss
        return loss


class LookLockerBaselineImagingCondition(LookLockerImagingCondition):
    def condition_loss(self, condition_input: Dict[str, torch.Tensor],
                       pred_orig_sample: torch.Tensor):
        """
        Perform reverse imaging on a single baseline image in MOLLI.

        :param condition_input: Conditional generation input.
        :param pred_orig_sample: estimated z0.
        :return: The conditional loss to guide the diffusion.
        """
        fa = condition_input["fa"]
        condition_input_scale = condition_input["input_scale"]
        z_init = self.z_init

        baseline = condition_input["xt"].to(pred_orig_sample) * condition_input_scale
        t_inv = condition_input["ti"]
        k, t1, t2 = self.rescale_prediction(pred_orig_sample, clip=True)
        xt = look_locker(k, fa, t1, t2, t_inv)
        reg = (pred_orig_sample - z_init.to(pred_orig_sample)).abs().mean(dim=(0, 2, 3))
        loss = self.baseline_distance(baseline, xt) + reg[0] * self.z_init_reg * 0.5 \
               + reg[1] * self.z_init_reg * 1.0 + reg[2] * self.z_init_reg * 0.5
        return loss
