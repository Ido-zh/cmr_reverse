import numpy as np
import torch
from typing import Tuple, Sequence


def padding_cropping(x: torch.Tensor, padding: Sequence[Tuple[int, int]],
                     cropping: Sequence[Tuple[int, int]]):
    """
    Perform the padding cropping operation on tensor.

    :param x: Tensor to be transformed.
    :param padding: Sequence of padding tuples [(dim1_left, dim1_right), (dim2_left, dim2_right)].
    :param cropping: Sequence of cropping tuples [(dim1_left, dim1_right), (dim2_left, dim2_right)].
    :return: Transformed x.
    """
    padding = (*padding[-1], *padding[-2])  # last dimension first.
    x = torch.nn.functional.pad(x, pad=padding)
    h, w = x.size()[-2:]
    x = x[..., cropping[0][0]:h - cropping[0][1], cropping[1][0]:w - cropping[1][1]]
    return x


def smart_padding_cropping(x,
                           target_size: Tuple[int, int]):
    """
    Automatically perform padding and cropping such that the tensor's shape matches the desired target shape.

    :param x: The tensor to be transformed.
    :param target_size: The target shape.

    :return: Tensor of the target size, padding and cropping tuples.
    """
    orig_size = x.shape[-2:]
    padding = []
    cropping = []
    for dim_idx in range(2):
        orig_ds, target_ds = orig_size[dim_idx], target_size[dim_idx]
        size_diff = target_ds - orig_ds  # positive for padding, negative for cropping
        if size_diff > 0:
            # padding
            padding.append((size_diff // 2, size_diff - size_diff // 2))
            cropping.append((0, 0))
        else:
            # cropping
            size_diff = - size_diff
            padding.append((0, 0))
            cropping.append((size_diff // 2, size_diff - size_diff // 2))
    x = padding_cropping(x, padding, cropping)
    return x, padding, cropping


class SquarePadResizeHelper:
    def __init__(self, target_size=128):
        self.target_size = target_size
        self.orig_shape = None
        self.pad = None
        self.crop = None
        self.square_size = None

    def transform(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        square_size = max(h, w)
        self.square_size = square_size

        square_image, pad, crop = smart_padding_cropping(x,
                                                         (square_size, square_size))
        self.pad = pad
        self.crop = crop

        # do interpolation
        image = torch.nn.functional.interpolate(square_image,
                                                size=(self.target_size, self.target_size),
                                                mode='bilinear', align_corners=True,
                                                antialias=True)
        return image

    def inv_transform(self, xt: torch.Tensor):
        x = torch.nn.functional.interpolate(xt,
                                            (self.square_size, self.square_size),
                                            align_corners=True,
                                            mode='bilinear', antialias=True)
        x = padding_cropping(x, self.crop, self.pad)
        return x

