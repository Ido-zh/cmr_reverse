import torch
import torch.nn.functional as F
import numpy as np
import math


def total_variation_isotropic(x, epsilon=1e-6):
    """
    Computes the isotropic Total Variation (TV) loss for a batch of images.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        epsilon (float): Small value to avoid sqrt(0).

    Returns:
        torch.Tensor: TV loss.
    """
    diff_x = x[:, :, :, :-1] - x[:, :, :, 1:]  # Difference along W
    diff_y = x[:, :, :-1, :] - x[:, :, 1:, :]  # Difference along H

    tv_loss = torch.sqrt(diff_x[:, :, :-1, :] ** 2 + diff_y[:, :, :, :-1] ** 2 + epsilon).mean()
    return tv_loss


def total_variation_anisotropic(x):
    """
    Computes the anisotropic Total Variation (TV) loss for a batch of images.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: TV loss.
    """
    diff_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # Difference along width (W)
    diff_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # Difference along height (H)

    tv_loss = diff_x.mean() + diff_y.mean()
    return tv_loss


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1, )
            padding = (pad_no, )
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = (cross * cross + 1e-5) / (I_var * J_var + 1e-5)

        return -torch.mean(cc, dim=(-2, -1))
