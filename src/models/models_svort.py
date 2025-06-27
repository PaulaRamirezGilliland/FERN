import random
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
from .transformer import SVRtransformer, SRRtransformer, SVRtransformerV2
from .reconstruction import PSFreconstruction, SRR
from transform import (
    RigidTransform,
    mat_update_resolution,
    ax_update_resolution,
    mat2axisangle,
    point2mat,
    mat2point,
)
from slice_acquisition import slice_acquisition, slice_acquisition_adjoint
import collections
from typing import Union
import torch.nn.functional as F



class SVoRT(nn.Module):
    def __init__(self, n_iter=3, iqa=True, vol=True, pe=True):
        super().__init__()
        self.n_iter = n_iter
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None

        svrnet_list = []
        for i in range(self.n_iter):
            svrnet_list.append(
                SVRtransformer(
                    n_res=50,
                    n_layers=4,
                    n_head=4 * 2,
                    d_in=9 + 2,
                    d_out=9,
                    d_model=256 * 2,
                    d_inner=512 * 2,
                    dropout=0.0,
                    res_d_in=4 if (i > 0 and self.vol) else 3,
                )
            )
        self.svrnet = nn.ModuleList(svrnet_list)
        if iqa:
            self.srrnet = SRRtransformer(
                n_res=34,
                n_layers=4,
                n_head=4,
                d_in=8,
                d_out=1,
                d_model=256,
                d_inner=512,
                dropout=0.0,
            )

    def forward(self, data):

        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "interp_psf": False,
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "s_thick": data["slice_thickness"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["seqs"]
        positions = data["positions"]

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0# + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        for i in range(self.n_iter):
            theta, attn = self.svrnet[i](
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = PSFreconstruction(mat, stacks, None, None, params)
                ax = mat2axisangle(_trans.matrix())
                ax = ax_update_resolution(ax, 1, params["res_s"])
            if self.iqa:
                volume, iqa_score = self.srrnet(
                    ax, mat, stacks, volume, params, positions
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)

        self.attn = attn.detach()

        return trans, volumes, thetas


def build_attn_mask(n_slices, fill_value, dtype, device):
    n_stack = len(n_slices)
    attn_mask = torch.zeros(
        (sum(n_slices) + n_stack, sum(n_slices) + n_stack), dtype=dtype, device=device
    )
    attn_mask[:, :n_stack] = fill_value
    i_slice = 0
    for i_stack, n_slice in enumerate(n_slices):
        attn_mask[i_stack, n_stack + i_slice : n_stack + i_slice + n_slice] = fill_value
        attn_mask[
            n_stack + i_slice : n_stack + i_slice + n_slice,
            n_stack + i_slice : n_stack + i_slice + n_slice,
        ] = fill_value
        i_slice += n_slice
    return attn_mask


class SVoRTv2(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True):
        super().__init__()
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None
        self.n_iter = n_iter

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=9 + 1, # set to 2 for stack + index
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1, 
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=9 + 1, # set to 2 for stack + index
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2 if self.vol else 1, 
        )

        if iqa:
            self.srr = SRR(n_iter=2)
    def forward(self, data):
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "volume_shape": data["volume_shape"],
        }
        
        transforms = RigidTransform(data["transforms"])
        stacks = data["seqs"]

        positions = data["positions"]
        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0
        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None

        for i in range(self.n_iter):
            svrnet = self.svrnet2 if i else self.svrnet1
            theta, iqa_score, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else data["STIC"].detach(),
                params,
            )

            thetas.append(theta)
            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)
            with torch.no_grad():

                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = slice_acquisition_adjoint(
                    mat,
                    params["psf"],
                    stacks,
                    None,
                    None,
                    params["volume_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    equalize=True,
                )
            if self.iqa:
                volume = self.srr(
                    mat,
                    stacks,
                    volume,
                    params,
                    iqa_score.view(-1, 1, 1, 1),
                    None,
                    None,
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
        self.attn = attn.detach()
        return trans, volumes, thetas


def gaussian_blur(
    x: torch.Tensor, sigma: Union[float, collections.abc.Iterable], truncated: float
) -> torch.Tensor:
    spatial_dims = len(x.shape) - 2
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = [sigma] * spatial_dims
    kernels = [gaussian_1d_kernel(s, truncated, x.device) for s in sigma]
    c = x.shape[1]
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
    for d in range(spatial_dims):
        s = [1] * len(x.shape)
        s[d + 2] = -1
        k = kernels[d].reshape(s).repeat(*([c, 1] + [1] * spatial_dims))
        padding = [0] * spatial_dims
        padding[d] = (k.shape[d + 2] - 1) // 2
        x = conv_fn(x, k, padding=padding, groups=c)
    return x

DeviceType = Union[torch.device, str, None]

# from MONAI
def gaussian_1d_kernel(
    sigma: float, truncated: float, device: DeviceType
) -> torch.Tensor:
    tail = int(max(sigma * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
    t = 0.70710678 / sigma
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return kernel.clamp(min=0)

