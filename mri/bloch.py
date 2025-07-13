"""
Signal models in magnetic resonance imaging, these describe the forward model of imaging that will be used in evaluating the likelihood.
    - For MOLLI, we adopt the TrueFISP imaging model in https://onlinelibrary.wiley.com/doi/10.1002/mrm.20058.
"""
import torch
import numpy as np


def safe_divide(a, b, tol=1e-5):
    c = a / torch.clamp(b, tol, None)
    return c


def steady_state_signal(proton_density, T1, T2, flip_angle):
    """
    Balanced steady-state free-precession model. This is an approximation assuming very short TR/TE.

    :param proton_density: The PD map, or magnetization strength map.
    :param T1: T1 map.
    :param T2: T2 map.
    :param flip_angle: Flip angle in RAD.
    :return:
    """
    sin_fa, cos_fa = np.sin(flip_angle), np.cos(flip_angle)
    S0 = proton_density * 1
    denominator = 1 + cos_fa + (1-cos_fa)*safe_divide(T1, T2)
    signal = S0 / denominator
    return signal


def inversion_factor(flip_angle, T1, T2):
    """
    Inversion factor in TrueFISP.

    :param flip_angle: Flip angle in RAD.
    :param T1: T1 map.
    :param T2: T2 map.
    :return: Inversion factor map.
    """
    ratio = safe_divide(T1, T2)
    inv = 1 + np.sin(flip_angle*0.5)/np.sin(flip_angle)*(ratio*(1-np.cos(flip_angle))+1+np.cos(flip_angle))
    return inv


def apparent_T1(flip_angle, T1, T2):
    """
    Apparent T1 map in TrueFISP.

    :param flip_angle: Flip angle in RAD.
    :param T1: T1 map.
    :param T2: T2 map.
    :return: Apparent T1 map.
    """
    T1app_inv = safe_divide(1, T1)*np.cos(flip_angle*0.5)**2 + safe_divide(1, T2)*np.sin(flip_angle*0.5)**2
    T1app = safe_divide(1, T1app_inv)
    return T1app


def look_locker(proton_density, flip_angle, T1, T2, inversion_time, phase_sensitive=False):
    """
    MOLLI signal equation.

    :param proton_density: PD map.
    :param flip_angle: FA in degrees.
    :param T1: T1 map.
    :param T2: T2 map.
    :param inversion_time: Inversion time.
    :param phase_sensitive: Phase sensitive keeps numbers with phases of 180 degrees (negative numbers).
    :return: The signal (image).
    """
    flip_angle = flip_angle * np.pi / 180
    ss = steady_state_signal(proton_density, T1, T2, flip_angle)
    inv = inversion_factor(flip_angle, T1, T2)
    T1app = apparent_T1(flip_angle, T1, T2)
    signal = ss * (1 - inv * torch.exp(-safe_divide(inversion_time, T1app)))
    if not phase_sensitive:
        signal = torch.abs(signal)
    return signal
