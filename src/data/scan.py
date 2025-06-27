import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Compose, GaussianSmooth, RandAffine, RandAdjustContrast
from slice_acquisition import slice_acquisition
from transform import RigidTransform, mat2point
from transform import random_angle, random_trans, RigidTransform, mat_update_resolution


def get_sparse_in(prob=0.3):
    """
    Returns a list of randomly selected slice indices based on predefined ranges,
    if 'test_transform' is not in `data` and with a certain probability - otherwise use full dense input.

    Parameters:
        prob (float): Probability of sparse standard view input. Default is 0.3.

    Returns:
        list or None: List of selected indices or None if condition not met.
    """


    if np.random.rand() > prob:
        return None

    random_slice_num = np.random.choice([1, 2, 3, 4])

    slice_ranges = {
        1: [[35, 45]],
        2: [[30, 35], [35, 45], [45, 56]],
        3: [[30, 35], [35, 40], [40, 45], [45, 56]],
        4: [[30, 35], [35, 40], [41, 45], [45, 50], [50, 56]],
    }

    indices_fg = [np.random.choice(np.arange(start, end)) for start, end in slice_ranges[random_slice_num]]

    return indices_fg



def get_PSF(
    r_max=None, res_ratio = (1, 1, 1), threshold=1e-3, device=torch.device("cpu")
):
    sigma_x = 1.2*res_ratio[0] / 2.3548
    sigma_y = 1.2*res_ratio[1] / 2.3548

    # Elevational res much worse in US
    sigma_z = 3.0*res_ratio[2] / 2.3548

    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    psf = torch.exp(
        -0.5
        * (
            grid_x**2 / sigma_x**2
            + grid_y**2 / sigma_y**2
            + grid_z**2 / sigma_z**2
        )
    )
    psf[psf.abs() < threshold] = 0
    rx = torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item()
    ry = torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item()
    rz = torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item()
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf


def init_seq_transforms(n_slice, gap, restricted, txy, device, identity=False):

    """
    Random training transformations applied to simulated slices
    Each slice gets an independent transformation
    Args:
        n_slice: (int) number of frames
        gap: (int) gap between frames (or elevational resolution for training)
        restricted: (bool) if True, sample slices close to standard view protocol. False = full 3D rotations
        txy: in-plane translations
        device: (torch.device) torch device
        identity: (bool) whether to apply identity transforms

    Returns: RigidTransform object containing training US sequence transforms

    """
    tz = (
        torch.arange(0, n_slice, device=device, dtype=torch.float32)
        - (n_slice - 1) / 2.0
    ) * gap

    tx = ty = torch.zeros_like(tz)

    if identity:
        angle = torch.zeros((n_slice, 3), dtype=torch.float).to(device)
    else:
        angle = random_angle(n_slice, restricted, device)

    if txy:
        tx = torch.FloatTensor(tz.shape).uniform_(-txy, txy).to(device)
        ty = torch.FloatTensor(tz.shape).uniform_(-txy, txy).to(device)

    t = torch.stack((tx, ty, tz), -1)
    return RigidTransform(torch.cat((angle, t), -1), trans_first=True)


def reset_transform(transform):
    transform = transform.axisangle()
    transform[:, :] = 0
    return RigidTransform(transform)


class Scanner:
    def __init__(self, kwargs):
        self.resolution_slice_fac = kwargs["resolution_slice_fac"]
        self.resolution_slice_max = kwargs["resolution_slice_max"]
        self.slice_thickness = kwargs["slice_thickness"]
        self.gap = kwargs["gap"]
        self.min_num_seq = kwargs["min_num_seq"]
        self.max_num_seq = kwargs["max_num_seq"]
        self.max_num_slices = kwargs.get("max_num_slices", None)
        self.noise_sigma = kwargs["noise_sigma"]
        self.slice_size = kwargs.get("slice_size", None)
        self.resolution_recon = kwargs.get("resolution_recon", None)
        self.restrict_transform = kwargs.get("restrict_transform", False)
        self.idx_views = kwargs.get("idx_views", [37, 41, 44, 53, 94])
        self.txy = kwargs.get("txy", 0)

    def get_resolution(self, data):
        resolution = self.resolution_recon
        if hasattr(self.resolution_slice_fac, "__len__"):
            resolution_slice = np.random.uniform(
                self.resolution_slice_fac[0] * resolution,
                min(
                    self.resolution_slice_fac[-1] * resolution,
                    self.resolution_slice_max,
                ),
            )
        else:
            resolution_slice = self.resolution_slice_fac * resolution
        if self.resolution_recon is not None:
            data["resolution_recon"] = self.resolution_recon
        else:
            data["resolution_recon"] = np.random.uniform(resolution, resolution_slice)
        data["resolution_slice"] = resolution_slice
        if hasattr(self.slice_thickness, "__len__"):
            data["slice_thickness"] = np.random.uniform(
                self.slice_thickness[0], self.slice_thickness[-1]
            )
        else:
            data["slice_thickness"] = self.slice_thickness
        if self.gap is None:
            data["gap"] = data["slice_thickness"]
        elif hasattr(self.gap, "__len__"):
            data["gap"] = np.random.uniform(
                self.gap[0], min(self.gap[-1], data["slice_thickness"])
            )
        else:
            data["gap"] = self.gap
        return data

    def add_noise(self, slices, threshold):
        if (not hasattr(self.noise_sigma, "__len__")) and self.noise_sigma == 0:
            return slices
        mask = slices > threshold
        masked = slices[mask]
        sigma = np.random.uniform(self.noise_sigma[0], self.noise_sigma[-1])
        noise1 = torch.randn_like(masked) * sigma
        noise2 = torch.randn_like(masked) * sigma
        slices[mask] = torch.sqrt((masked + noise1) ** 2 + noise2**2)
        return slices

    def signal_void(self, slices):

        idx = torch.rand(slices.shape[0], device=slices.device) < self.prob_void
        n = idx.sum()
        if n > 0:
            h, w = slices.shape[-2:]
            y = torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=slices.device)
            x = torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=slices.device)
            yc = (torch.rand(n, device=slices.device) - 0.5) * (h - 1)
            xc = (torch.rand(n, device=slices.device) - 0.5) * (w - 1)

            y = y.view(1, -1, 1) - yc.view(-1, 1, 1)
            x = x.view(1, 1, -1) - xc.view(-1, 1, 1)

            theta = 2 * np.pi * torch.rand((n, 1, 1), device=slices.device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            x, y = c * x - s * y, s * x + c * y

            a = 5 + torch.rand_like(theta) * 10
            A = torch.rand_like(theta) * 0.5 + 0.5
            sx = torch.rand_like(theta) * 15 + 1
            sy = a**2 / sx

            sx = -0.5 / sx**2
            sy = -0.5 / sy**2

            mask = 1 - A * torch.exp(sx * x**2 + sy * y**2)
            slices[idx, 0] *= mask
        return slices


    # Coarse view Positional embedding
    def coarse_pe(self, ns, indices_fg):
        """

        Args:
            ns: (int) number of frames in training seq.
            indices_fg: (list) indices of foreground frames in training seq.

        Returns: (torch.tensor) containing positional indices for slices sampled

        """

        pos_all = np.zeros(ns)
        # Set positions for different views
        ind_start = 0
        for ind, ind_end in enumerate(self.idx_views):
            pos_all[ind_start:ind_end] = ind
            ind_start = ind_end

        id_pos = torch.tensor(pos_all[indices_fg]).unsqueeze(1).float()
        return id_pos


    def scan(self, data):
        data = self.get_resolution(data)
        res = data["resolution_recon"]
        res_r = data["resolution_recon"]
        res_s = data["resolution_slice"]
        s_thick = data["slice_thickness"]
        gap = data["gap"]
        device = data["volume"].device
        if res_r != res:
            grids = []
            for i in range(3):
                size_new = int(data["volume"].shape[i + 2] * res / res_r)
                grid_max = (
                    (size_new - 1) * res_r / (data["volume"].shape[i + 2] - 1) / res
                )
                grids.append(
                    torch.linspace(-grid_max, grid_max, size_new, device=device)
                )
            grid = torch.stack(
                torch.meshgrid(*grids, indexing="ij")[::-1], -1
            ).unsqueeze_(0)
            volume_gt = F.grid_sample(data["volume"], grid, align_corners=True)
        else:
            volume_gt = data["volume"].clone()
        data["volume_gt"] = volume_gt

        psf_acq = get_PSF(
            res_ratio=(res_s / res, res_s / res, s_thick / res), device=device
        )
        psf_rec = get_PSF(
            res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r), device=device
        )
        data["psf_rec"] = psf_rec
        data["psf_acq"] = psf_acq

        vs = data["volume"].shape[2]
        if self.slice_size is None:
            ss = int(
                np.sqrt((vs[-1] ** 2 + vs[-2] ** 2 + vs[-3] ** 2) / 2.0) * res / res_s
            )
            ss = int(np.ceil(ss / 32.0) * 32)
        else:
            ss = self.slice_size
        ns = int(vs * res / gap)

        seqs, seqs_mask = [], []
        transforms, transforms_gt = [], []
        positions_all = []

        max_num_seq = np.random.randint(self.min_num_seq, self.max_num_seq + 1)
        slices_mask = None

        while True:
            # 2D test time - do not apply simulated transforms (inference)
            if "test_transform" in data:

                # TODO - get rid of this
                sigma_x = 1.2 / 2.3548
                sigma_y = 1.2 / 2.3548
                sigma_z = 3*1 / 2.3548

                # Shape N_frames, C, H, W
                smoother = GaussianSmooth(sigma=[sigma_z, sigma_y, sigma_x], approx="erf")
                # Applying a smoothing filter
                slices = (smoother(data['img_2d'].squeeze(0))).squeeze()[:].unsqueeze(1)

                ns = slices.shape[0]
                transform_target = init_seq_transforms(
                    ns, gap, self.restrict_transform, 0, device, identity=True
                )

            # Simulate transformations from volume
            else:
                # seq transformation
                transform_target = init_seq_transforms(
                   ns , gap, self.restrict_transform, self.txy, device
               )

                mat = mat_update_resolution(transform_target.matrix(), 1, res)

                slices = slice_acquisition(
                    mat,
                    data["volume"],
                    None,
                    None,
                    psf_acq,
                    (ss, ss),
                    res_s / res,
                    False,
                    False,
                    )

                if "mask_heart" in data:
                    slices_mask = slice_acquisition(
                        mat,
                        data["mask_heart"],
                        None,
                        None,
                        psf_acq,
                        (ss, ss),
                        res_s / res,
                        False,
                        False,
                    ).int()

            # Get rid of background/empty slices
            indices_fg = []
            for ind_s, i in enumerate(slices):
                if i.sum() == 0:
                    continue
                else:
                    indices_fg.append(ind_s)

            # Randomly switch to 1-5 standard views with a 30%
            # TODO - comment this back out
            # if not "test_transform" in data:
            #     sparse_indices = get_sparse_in(prob=0.3)
            #     if sparse_indices:
            #         indices_fg = sparse_indices

            if slices_mask:
                slices_mask = slices_mask[indices_fg]
            slices = slices[indices_fg]

            transform_target = transform_target[indices_fg]
            transform_init = transform_target.clone()
            transform_init = reset_transform(transform_init)

            if "PE_2D" not in data:
                positions = self.coarse_pe(ns, indices_fg)

            else:
                positions = torch.tensor(data["PE_2D"][:]).float()
                if isinstance(data["PE_2D"], list):
                    positions = positions.unsqueeze(1)

            positions = positions[indices_fg]
            # Augmentation
            if "test_transform" in data:
                print("Testing (no aug)")
            else:
                slices = self.add_noise(slices, 0)
                tf = Compose([RandAdjustContrast(prob=0.3, gamma=(0.5, 1.5)),
                          RandAffine(prob=0.2, shear_range=(0.1, 0.2), scale_range=(-0.9, 0.2))])

                slices = tf(slices)

            if (
                self.max_num_slices is not None
                and sum(st.shape[0] for st in seqs) + slices.shape[0]
                >= self.max_num_slices
            ):
                break
            seqs.append(slices)
            if slices_mask:
                seqs_mask.append(slices_mask)
            transforms.append(transform_init)
            transforms_gt.append(transform_target)
            positions_all.append(positions)

            if len(seqs) >= max_num_seq:
                break

        if "test_transform" not in data:
            data.pop("volume")
            if "atlas" in data:
                data.pop("atlas")
            if "img_2d" in data:
                data.pop("img_2d")

        seqs = torch.cat(seqs, 0)
        if slices_mask:
            seqs_mask = torch.cat(seqs_mask, 0)
        transforms = RigidTransform.cat(transforms)
        transforms_gt = RigidTransform.cat(transforms_gt)
        positions_all = torch.cat(positions_all, dim=0)

        data["slice_shape"] = (ss, ss)
        data["volume_shape"] = volume_gt.shape[-3:]
        data["seqs"] = seqs
        data["seqs_mask"] = seqs_mask
        data["positions"] = positions_all
        data["transforms"] = transforms.matrix()
        data["transforms_gt"] = transforms_gt.matrix()

        return data
