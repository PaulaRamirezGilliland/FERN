import torch
import torch.nn as nn
import torch.nn.functional as F
from transform import axisangle2mat
from slice_acquisition import slice_acquisition, slice_acquisition_adjoint
from typing import Optional, Dict


def dot(x, y):
    return torch.dot(x.flatten(), y.flatten())

def CG(A, b, x0, n_iter):
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new

def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params):
    return slice_acquisition_adjoint(transforms, params['psf'], slices, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], True)
    

class SRR(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.output_relu = output_relu

    def forward(
        self,
        transforms: torch.Tensor,
        slices: torch.Tensor,
        volume: torch.Tensor,
        params: Dict,
        p: Optional[torch.Tensor] = None,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        At = lambda x: self.At(transforms, x, params, slices_mask, vol_mask)
        AtA = lambda x: self.AtA(transforms, x, p, params, slices_mask, vol_mask)

        b = At(slices * p if p is not None else slices)
        volume = CG(AtA, b, volume, self.n_iter)

        if self.output_relu:
            return F.relu(volume, True)
        else:
            return volume

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        p: Optional[torch.Tensor],
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        slices = self.A(transforms, x, params, slices_mask, vol_mask)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, params, slices_mask, vol_mask)
        return vol


