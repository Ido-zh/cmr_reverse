import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from .utils import smart_padding_cropping, padding_cropping


class NiftiImage:
    def __init__(self, filepath, quantile=(0, 98)):
        self.file = Path(filepath)
        self.identifier = self.file.name[:-7][:-5]
        self.nifti = nib.load(self.file)
        self.image = self.nifti.get_fdata()  # (H, W, S)
        h, w, s = self.image.shape

        amin, amax = np.percentile(self.image, q=quantile, axis=(0, 1), keepdims=True)
        image = (self.image - amin) / (amax - amin)
        self.image = np.clip(image, 0., 1.)
        self.square_size = max(h, w)

        self.image_tensor = torch.from_numpy(self.image)
        self.image_tensor = self.image_tensor.unsqueeze(0)  # (1, H, W, S)
        self.image_tensor = self.image_tensor.permute((3, 0, 1, 2))  # (S, 1, H, W)

        if h < w:
            self.transpose = True
        else:
            self.transpose = False

        self.pad = None
        self.crop = None
        self.slices = s

    def get_condition(self, sl, cond_size=128, fa=45, in_scale=0.3, TR=35, TE=15):
        square_size = self.square_size
        if self.transpose:
            image = self.image_tensor.permute((0, 1, 3, 2))
        else:
            image = self.image_tensor

        image = image[[sl], ...]

        # pad/crop to square
        square_image, pad, crop = smart_padding_cropping(image, (square_size, square_size))
        self.pad = pad
        self.crop = crop

        # do interpolation
        image = torch.nn.functional.interpolate(square_image, size=(cond_size, cond_size),
                                                mode='bilinear', align_corners=True,
                                                antialias=True)

        return dict(ssfp=image,
                    # init=init,
                    fa=fa * np.pi / 180.,
                    TR=TR,
                    TE=TE,
                    input_scale=in_scale)

    def convert_hidden(self, hidden):
        """
        Convert hidden back to the proper shape.

        :param hidden: Tensor of shape [S, 3, H, W]
        :return:
        """
        hidden = torch.nn.functional.interpolate(hidden,
                                                 (self.square_size, self.square_size),
                                                 align_corners=True,
                                                 mode='bilinear', antialias=True)
        hidden = padding_cropping(hidden, self.crop, self.pad)

        if self.transpose:
            hidden = torch.permute(hidden, (0, 1, 3, 2))
        return hidden


class NiftiImageLabel:
    def __init__(self, filepath, label_path, quantile=(0, 99)):
        self.file = Path(filepath)
        self.label_file = Path(label_path)
        self.identifier = self.file.name[:-7][:-5]
        self.nifti = nib.load(self.file)
        self.image = self.nifti.get_fdata()  # (H, W, S)
        self.label = nib.load(self.label_file).get_fdata()
        self.quantile = quantile

        h, w, s = self.image.shape

        amin, amax = np.percentile(self.image, q=quantile, axis=(0, 1), keepdims=True)
        image = (self.image - amin) / (amax - amin)
        self.image = np.clip(image, 0., 1.)
        self.square_size = max(h, w)

        self.image_tensor = torch.from_numpy(self.image)
        self.image_tensor = self.image_tensor.unsqueeze(0)  # (1, H, W, S)
        self.image_tensor = self.image_tensor.permute((3, 0, 1, 2))  # (S, 1, H, W)
        self.label_tensor = torch.from_numpy(self.label).long()
        self.label_tensor = torch.nn.functional.one_hot(self.label_tensor, num_classes=4)
        self.label_tensor = self.label_tensor.permute((2, 3, 0, 1)).float()

        if h < w:
            self.transpose = True
        else:
            self.transpose = False

        self.pad = None
        self.crop = None
        self.slices = s

    def get_condition(self, sl, cond_size=128, fa=45, in_scale=0.3, TR=35, TE=15):
        square_size = self.square_size
        if self.transpose:
            image = self.image_tensor.permute((0, 1, 3, 2))
            label = self.label_tensor.permute((0, 1, 3, 2))
        else:
            image = self.image_tensor
            label = self.label_tensor

        image = image[[sl], ...]
        label = label[[sl], ...]

        # pad/crop to square
        square_image, pad, crop = smart_padding_cropping(image, (square_size, square_size))
        square_label, _, _ = smart_padding_cropping(label, (square_size, square_size))
        self.pad = pad
        self.crop = crop

        # do interpolation
        image = torch.nn.functional.interpolate(square_image, size=(cond_size, cond_size),
                                                mode='bilinear', align_corners=True,
                                                antialias=True)
        label = torch.nn.functional.interpolate(square_label, size=(cond_size, cond_size),
                                                mode='bilinear', align_corners=True,
                                                antialias=True)
        label = torch.argmax(label, dim=1, keepdim=True)
        fg = (label > 0).float()
        myo = (label == 2).float()
        blood = fg - myo
        mask = torch.cat([myo, blood], dim=1)
        control = torch.cat([image*2-1, mask], dim=1)

        return dict(ssfp=image,
                    control=control,
                    fa=fa * np.pi / 180.,
                    TR=TR,
                    TE=TE,
                    input_scale=in_scale)

    def convert_hidden(self, hidden):
        """
        Convert hidden back to the proper shape.

        :param hidden: Tensor of shape [S, 3, H, W]
        :return:
        """
        hidden = torch.nn.functional.interpolate(hidden,
                                                 (self.square_size, self.square_size),
                                                 align_corners=True,
                                                 mode='bilinear', antialias=True)
        hidden = padding_cropping(hidden, self.crop, self.pad)

        if self.transpose:
            hidden = torch.permute(hidden, (0, 1, 3, 2))
        return hidden

