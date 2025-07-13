"""
This script defines a few log-likelihood classes. The log-likelihood is a synonym for fidelity loss, data consistency loss, et, they can appear as L2 norm (Gaussian), L1 norm (Laplacian) etc.
"""
import torch
from typing import Dict, Tuple, Union, Sequence


class Likelihood:
    """Abstract class for evaluating the log-likelihood"""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class GaussianLikelihood(Likelihood):
    """ Gaussian likelihood leads to L2 norm. """
    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor):
        d = (y.to(y_pred) - y_pred) ** 2 / self.sigma ** 2
        return d.mean()


class LaplacianLikelihood(Likelihood):
    """ Laplacian likelihood leads to L1 norm. """
    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor):
        d = (y.to(y_pred) - y_pred).abs() / self.sigma
        return d.mean()


class RayleighLikelihood(Likelihood):
    """ Rayleigh likelihood. """
    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor):
        residual = y.to(y_pred) - y_pred
        d = 0.5 * residual.square() / self.sigma ** 2 - torch.log(torch.clamp(residual, 1e-3, None))
        return d.mean()


def get_likelihood(sigma, likelihood: str = 'gaussian') -> Likelihood:
    """
    Get a likelihood object.

    :param sigma: STD of the distribution.
    :param likelihood: Type of the likelihood, can be 'gaussian', 'laplacian', or 'rayleigh'.
    :return: The log-likelihood evaluator.
    """
    if likelihood == "gaussian":
        distance = GaussianLikelihood(sigma)
    elif likelihood == "laplacian":
        distance = LaplacianLikelihood(sigma)
    elif likelihood == "rayleigh":
        distance = RayleighLikelihood(sigma)
    else:
        raise ValueError(f"Unknown likelihood {likelihood}.")
    return distance


class ImagingCondition:
    def __init__(self, rho_rescale: float = 0.6,
                 t1_rescale: float = 2800.,
                 t2_rescale: float = 350.,
                 sigma: float = 0.12,
                 likelihood: str = "gaussian"):
        """


        :param rho_rescale: Rescaler for diffusion model outputs, rho = z[0] / rho_rescale.
        :param t1_rescale: T1 = z[1] * t1_rescale
        :param t2_rescale: T2 = z[2] * t2_rescale
        :param sigma: STD of the likelihood distribution.
        :param likelihood: Likelihood type.
        """
        self.rho_rescaler = rho_rescale
        self.t1_rescaler = t1_rescale
        self.t2_rescaler = t2_rescale

        self.likelihood = likelihood.lower()
        self.sigma = sigma
        self.distance = get_likelihood(self.sigma, self.likelihood)

    def rescale_prediction(self, pred_original_sample: torch.Tensor, clip=False):
        """
        Rescales the Diffusion model outputs z to real PD, T1, and T2 scales.

        :param pred_original_sample: Usually the z0 estimate using Tweedie formula in DDPM, it is also named so in the diffuser DDPM pipelines.
        :param clip: Clip the outputs such that the estimated PD, T1, T2 are positive numbers.
        :return: PD [a.u.], T1 [ms], and T2 [ms] in realistic scales.
        """
        if clip:
            pred_original_sample = torch.clamp(pred_original_sample, -1., None)
        k = (pred_original_sample[:, [0], ...] + 1) / 2 / self.rho_rescaler
        t1 = (pred_original_sample[:, [1], ...] + 1) / 2 * self.t1_rescaler
        t2 = (pred_original_sample[:, [2], ...] + 1) / 2 * self.t2_rescaler
        return k, t1, t2

    def condition_loss(self, condition_input: Dict[str, torch.Tensor],
                       pred_orig_sample: torch.Tensor):
        raise NotImplementedError

