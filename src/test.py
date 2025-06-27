import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics import MultiScaleSSIMMetric, PSNRMetric
from monai.transforms import CropForeground, SpatialCrop
from monai.transforms import LabelToContour
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score

from config import get_config
from data.dataset import EchoDataset
from data.io import save_volume
from data.scan import Scanner
from slice_acquisition import slice_acquisition_adjoint
from transform import mat2point, RigidTransform
from models import *


def get_useful_slices(volume):
    indices_fg = []
    for ind_s in range(volume.shape[2]):
        if volume[0,0,ind_s,...].sum() == 0:
            continue
        else:
            indices_fg.append(ind_s)

    volume = [volume[:,:,i,...] for i in indices_fg]
    volume = torch.cat(volume)
    return volume


def compute_psnr(original, reconstructed, data_range=None):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two 2D images.

    Args:
        original (np.ndarray): The original image.
        reconstructed (np.ndarray): The reconstructed or compared image.
        data_range (float or None): The range of the pixel values. If None, it is set to max(original.max(), reconstructed.max()).

    Returns:
        float: Peak Signal-to-Noise Ratio score.
    """
    # Ensure that the images are of the same shape
    if original.shape != reconstructed.shape:
        raise ValueError("Input images must have the same dimensions.")
    # Convert to NumPy arrays if necessary
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    # Compute Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)

    if data_range is None:
        data_range = np.max(original) - np.min(original)

    # Compute PSNR
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr



def compute_combined_foreground_mask(pred_slice, target_slice):
    foreground_mask_pred = pred_slice > 0
    foreground_mask_target = target_slice > 0
    
    # AND operation (overlap)
    foreground_mask_and = foreground_mask_pred & foreground_mask_target
    
    # OR operation (union)
    foreground_mask_or = foreground_mask_pred | foreground_mask_target
    
    # Combine the AND and OR masks (keeping all foreground regions)
    foreground_mask_combined = foreground_mask_or.clone()
    foreground_mask_combined[foreground_mask_and] = True
    
    return foreground_mask_combined


def compute_bounding_box(foreground_mask):
    nonzero_indices = torch.nonzero(foreground_mask)
    min_indices = nonzero_indices.min(dim=0)[0]
    max_indices = nonzero_indices.max(dim=0)[0] + 1  # +1 because slice end is exclusive

    return min_indices, max_indices

def threshold_at_zero(x):
    return x > 0

def crop_to_foreground(pred_slice, target_slice, foreground_mask):

    pred_slice = pred_slice.unsqueeze(0)
    target_slice = target_slice.unsqueeze(0)
    foreground_mask = foreground_mask.unsqueeze(0)

    cropper = CropForeground(select_fn=threshold_at_zero, return_coords=True)
    out = cropper(foreground_mask)

    tf_crop = SpatialCrop(roi_start = out[1], roi_end = out[2])
    pred_cropped = tf_crop(pred_slice)[0,...]
    target_cropped = tf_crop(target_slice)[0,...]

    return pred_cropped, target_cropped


def compute_mutual_information(pred_slice, target_slice, bins=64):
    # Flatten the slices
    pred_flat = pred_slice.cpu().numpy().ravel()
    target_flat = target_slice.cpu().numpy().ravel()
    
    # Compute 2D histogram
    hist_2d, _, _ = np.histogram2d(pred_flat, target_flat, bins=bins)
    
    # Compute mutual information
    mi = mutual_info_score(None, None, contingency=hist_2d)
    
    return mi



def compute_nmi(image1, image2, bins=64):
    """
    Compute the Normalized Mutual Information (NMI) between two 2D images.

    Args:
        image1 (np.ndarray or torch.Tensor): The first image.
        image2 (np.ndarray or torch.Tensor): The second image.
        bins (int): Number of bins for histogram computation.

    Returns:
        float: Normalized Mutual Information score.
    """
    # Convert to NumPy arrays if necessary
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().numpy()
    
    # Flatten the images to create 1D arrays
    image1_flat = image1.ravel()
    image2_flat = image2.ravel()

    # Compute the joint histogram
    hist_2d, _, _ = np.histogram2d(image1_flat, image2_flat, bins=bins)

    # Compute joint probability distribution
    pxy = hist_2d / np.sum(hist_2d)  # Joint probability distribution

    # Compute marginal probabilities
    px = np.sum(pxy, axis=1)  # Marginal probability for image1
    py = np.sum(pxy, axis=0)  # Marginal probability for image2

    # Compute entropies
    Hx = -np.sum(px * np.log(px + 1e-10))  # Entropy of image1
    Hy = -np.sum(py * np.log(py + 1e-10))  # Entropy of image2
    Hxy = -np.sum(pxy * np.log(pxy + 1e-10))  # Joint entropy

    # Compute mutual information
    mi = Hx + Hy - Hxy

    # Compute normalized mutual information
    nmi = 2 * mi / (Hx + Hy)

    return nmi

def compute_normalized_cross_correlation(pred_slice, target_slice):
    
    # Compute mean for each slice
    pred_mean = pred_slice.mean()
    target_mean = target_slice.mean()

    # Compute the numerator and denominator for NCC
    num = ((pred_slice - pred_mean) * (target_slice - target_mean)).sum()
    den = torch.sqrt(((pred_slice - pred_mean) ** 2).sum() * ((target_slice - target_mean) ** 2).sum())

    # Compute NCC
    ncc = num / den

    return ncc.item()


def compute_local_normalized_cross_correlation(pred_slice, target_slice, window_size=7, stride=1):
    # Calculate padding
    padding = window_size // 2

    # Extract patches
    pred_patches = F.unfold(pred_slice.unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=stride, padding=padding)
    target_patches = F.unfold(target_slice.unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=stride, padding=padding)

    pred_patches = pred_patches.view(window_size * window_size, -1)
    target_patches = target_patches.view(window_size * window_size, -1)

    # Compute mean for each patch
    pred_mean = pred_patches.mean(dim=0, keepdim=True)
    target_mean = target_patches.mean(dim=0, keepdim=True)

    # Compute NCC
    num = ((pred_patches - pred_mean) * (target_patches - target_mean)).mean(dim=0)
    den = torch.sqrt(((pred_patches - pred_mean) ** 2).mean(dim=0) * ((target_patches - target_mean) ** 2).mean(dim=0))

    ncc = (num / den).mean().item()

    return ncc


def compute_ssim(pred_slice, target_slice, data_range=1):
    pred_slice_np = pred_slice.cpu().numpy()
    target_slice_np = target_slice.cpu().numpy()

    # Compute data range if not provided
    if data_range is None:
        data_range = max(pred_slice_np.max(), target_slice_np.max()) - min(pred_slice_np.min(), target_slice_np.min())

    # Compute SSIM
    ssim_value = ssim(pred_slice_np, target_slice_np, data_range=data_range, win_size=11)

    return ssim_value


def process_slices(y_pred, y_target, mask=False):
    B, C, H, W = y_pred.shape
    mutual_info_slices, ncc_slices, ssim_slices, psnr_slices = [], [], [], []
    metric_ssim = MultiScaleSSIMMetric(spatial_dims=2, reduction="mean", weights=(0.0448, 0.2856), kernel_size=2)
    metric_psnr = PSNRMetric(max_val=1, reduction="mean")
    for b in range(B):
        for c in range(C):
            pred_slice = y_pred[b, c]
            target_slice = y_target[b, c]
            pred_cropped = pred_slice
            target_cropped = target_slice
            if mask:
                # Compute the combined foreground mask
                foreground_mask_combined = compute_combined_foreground_mask(pred_slice, target_slice)
                #Crop slices to the foreground region
                pred_cropped, target_cropped = crop_to_foreground(pred_slice, target_slice, foreground_mask_combined)
            # Compute metrics
            mi = compute_nmi(pred_cropped, target_cropped)
            #psnr = compute_psnr(pred_cropped, target_cropped, data_range=1)
            psnr = metric_psnr(pred_cropped.unsqueeze(0).unsqueeze(0), target_cropped.unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()[0][0]

            ncc = compute_normalized_cross_correlation(pred_cropped, target_cropped)
            ssim = metric_ssim(pred_cropped.unsqueeze(0).unsqueeze(0), target_cropped.unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()[0][0]
            # Append metrics to the list
            mutual_info_slices.append(mi)
            ncc_slices.append(ncc)
            ssim_slices.append(ssim)
            psnr_slices.append(psnr)

    return mutual_info_slices, ncc_slices, ssim_slices, psnr_slices  


class NCC:
    """
    NCC loss
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, pred_img, target_img, eps=1e-5):
        y_true_dm = target_img - torch.mean(target_img)
        y_pred_dm = pred_img - torch.mean(pred_img)

        ncc_num = torch.sum(y_true_dm * y_pred_dm).pow(2)
        ncc_den = torch.sum(torch.pow(y_true_dm, 2)) * torch.sum(torch.pow(y_pred_dm, 2)) + eps

        return -torch.div(ncc_num, ncc_den)

def compute_rmse(t_a, t_b):
    # Compute the squared differences
    squared_diffs = (t_a - t_b) ** 2

    # Mean squared error across the 3D components for each slice
    mse_per_slice = torch.mean(squared_diffs, dim=1)

    # Compute the RMSE for each slice by taking the square root
    rmse_per_slice = torch.sqrt(mse_per_slice)
    return rmse_per_slice


def compute_ed(point_pred, point_gt):
    if isinstance(point_pred, list):
        dist_all = []
        for slice_ind in range(len(point_pred)):
            #dist_slice = [np.linalg.norm(point_pred[slice_ind][point, :] - point_gt[slice_ind][point, :]) for point in
            #         range(3)]
            if len(point_pred[slice_ind].shape)==1:
                dist_slice = [np.linalg.norm(point_pred[slice_ind] - point_gt[slice_ind])]

            elif point_pred[slice_ind].shape[1]==3:
                dist_slice = [np.linalg.norm(point_pred[slice_ind] - point_gt[slice_ind], axis=1)]
            else:
                dist_slice = [np.linalg.norm(point_pred[slice_ind] - point_gt[slice_ind], axis=0)]
            dist_all.append(np.mean(dist_slice))

    else: 
        # Reshape points
        pts_pred = point_pred.reshape(point_pred.shape[0], 3, -1).cpu()
        pts_gt = point_gt.reshape(point_gt.shape[0], 3, -1).cpu()
        dist_all = []
        for slice_ind in range(pts_pred.shape[0]):
            dist_slice = [np.linalg.norm(pts_pred[slice_ind, point, :] - pts_gt[slice_ind, point, :]) for point in
                     range(3)]
            dist_all.append(np.mean(dist_slice))

    return dist_all


def compute_gd(transforms, transforms_gt):
    transforms_err = transforms_gt.inv().compose(transforms).matrix()
    transforms_err = transforms_err[..., :-1]

    gd_all = []
    for slice in range(transforms_err.shape[0]):
        gd_all.append(np.arccos((np.trace(transforms_err[slice, ...].cpu())-1)/2))
    return gd_all


# Loop over all slices
def contour2points(mask, mat, sx, sy, rs):

    mask = mask.cpu()
    mat = mat.cpu()
    all_points = []
    used_slices = []
    mask = mask.squeeze(1) 
    # Loop through slices
    for slice_ind in range(mask.shape[0]):
        mask_slice = mask[slice_ind, ...]
        if torch.sum(mask_slice) == 0:
            continue
        mat_slice = mat[slice_ind, ...].unsqueeze(0)
        # Apply LabelToContour transform
        label_to_contour = LabelToContour()
        contour_map = label_to_contour(mask_slice.unsqueeze(0).float())
        contour_points = torch.argwhere(contour_map[0, ...] == 1)
        if contour_points.shape[0]<150:
            continue
        contour_points = torch.cat((contour_points, torch.zeros_like(contour_points[..., 0:1])), dim=1)
        # Normalize and prepare points
        p = (contour_points - torch.tensor([(sx - 1) / 2, (sy - 1) / 2, 0], dtype=mat_slice.dtype, device=mat_slice.device)) * rs
        p = p.unsqueeze(0).unsqueeze(-1)

        # Apply rotations and translations
        R = mat_slice[:, :, :-1].unsqueeze(1) # nx1x3x3
        T = mat_slice[:, :, -1:].unsqueeze(1) # nx1x3x1
        p = torch.matmul(R, p + T)
        p = p.squeeze()#.T  # Nx3
        used_slices.append(slice_ind)
        all_points.append(p)

    return all_points, used_slices  # .view(-1, -1)

def mask2points(mask, mat, used_slices, sx, sy, rs):

    all_points = []
    mask = mask.cpu()
    mat = mat.cpu()
    mask = mask.squeeze(1) 
# Loop through slices
    for slice_ind in used_slices:
        mask_slice = mask[slice_ind, ...]
        if torch.sum(mask_slice) == 0:
            continue
        mat_slice = mat[slice_ind, ...].unsqueeze(0)

        # Get all points in the mask (non-zero locations) 
        mask_points = torch.argwhere(mask_slice > 0)
        mask_points = torch.cat((mask_points, torch.zeros_like(mask_points[..., 0:1])), dim=1)
        # Normalize and prepare points
        p = (mask_points - torch.tensor([(sx - 1) / 2, (sy - 1) / 2, 0], dtype=mat_slice.dtype, device=mat_slice.device)) * rs
        p = p.unsqueeze(0).unsqueeze(-1)

        # Apply rotations and translations
        R = mat_slice[:, :, :-1].unsqueeze(1) # nx1x3x3
        T = mat_slice[:, :, -1:].unsqueeze(1) # nx1x3x1
        p = torch.matmul(R, p + T)
        p = p.squeeze()#.T  # Nx3
        all_points.append(p)

    return all_points


if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to the yaml config file", required=True, type=str
    )
    parser.add_argument(
        "--checkpoint", help="path to the checkpoint file", required=True, type=str
    )
    parser.add_argument("--output", help="output path", required=True, type=str)


    args = parser.parse_args()


    cfg = get_config(args.config)
    # mkdir
    os.makedirs(args.output, exist_ok=True)
    # model
    device = torch.device(cfg["model"]["device"])
    model = globals()[cfg["model"]["model_type"]](**cfg["model"]["model_param"])
    cp = torch.load(args.checkpoint)
    model.to(device)
    model.load_state_dict(cp["model"])
    model.eval()
    print("Iter", cp["iter"])
    # test dataset
    dataset = EchoDataset(True, False, cfg["dataset"])
    scanner = Scanner(cfg["scanner"])

    for i in range(len(dataset)):
        # read data
        data = dataset.get_data()
        data = scanner.scan(data)
        for k in data:
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(device, non_blocking=True)
        path_save = os.path.join(args.output, str(i))
        os.makedirs(path_save, exist_ok=True)
        # run models
        transforms = {}
        volumes = {}
        points = {}

        with torch.no_grad():
            transforms, volumes, points = model(data)

        mat = transforms[-1].matrix()
        if data["seqs_mask"]:
            contour_points_gt, used_slices = contour2points(data["seqs_mask"],
                                            RigidTransform(data["transforms_gt"]).inv().matrix(),
                                            data["slice_shape"][0],
                                            data["slice_shape"][1],
                                            data["resolution_slice"]
                                            )

            mask_points_gt = mask2points(data["seqs_mask"],
                                            RigidTransform(data["transforms_gt"]).inv().matrix(),
                                            used_slices,
                                            data["slice_shape"][0],
                                            data["slice_shape"][1],
                                            data["resolution_slice"]
                                            )


            contour_points_pred, _ = contour2points(data["seqs_mask"],
                                            RigidTransform(mat).inv().matrix(),
                                            data["slice_shape"][0],
                                            data["slice_shape"][1],
                                            data["resolution_slice"]
                                            )

            mask_points_pred = mask2points(data["seqs_mask"],
                                            RigidTransform(mat).inv().matrix(),
                                            used_slices,
                                            data["slice_shape"][0],
                                            data["slice_shape"][1],
                                            data["resolution_slice"]
                                            )

            # Get GT points
            points_gt = mat2point(
                data["transforms_gt"],
                data["slice_shape"][0],
                data["slice_shape"][1],
                data["resolution_slice"])


            # Compute ED and GD
            ED_central = compute_ed(contour_points_pred, contour_points_gt)
            ED_mask = compute_ed(mask_points_pred, mask_points_gt)

            ED = compute_ed(points[-1], points_gt)
            print("ED ", ED)
            print("ED central", ED_central)
            print("ED mask", ED_mask)
            GD = compute_gd(transforms[-1], RigidTransform(data["transforms_gt"]))
            GD_deg = [i*180/np.pi for i in GD]
            print("GD Deg = ", GD_deg)

            axisangle_gt = RigidTransform(data["transforms_gt"]).axisangle()
            axisangle_pred = RigidTransform(mat).axisangle()

            rmse_t = compute_rmse(axisangle_gt[:, 3:], axisangle_pred[:, 3:])
            print("RMSE T", rmse_t)
            df = pd.DataFrame()

            ED = [ED[used_ind] for used_ind in used_slices]
            GD_deg = [GD_deg[used_ind] for used_ind in used_slices]
            df['ED (mm)'] = ED
            df['ED contour (mm)'] = ED_central
            df['ED mask (mm)'] = ED_mask
            df['GD (deg)'] = GD_deg
            df['RMSE trans (mm)'] = rmse_t.cpu()
            print(" ")
     

        
        volume_pred = slice_acquisition_adjoint(
                    mat,
                    data["psf_rec"],
                    data['seqs'],
                    None,
                    None,
                    data["volume_shape"],
                    data["resolution_slice"] / data["resolution_recon"],
                    False,
                    equalize=True,
                )

        # save transform
        np.save(
             os.path.join(path_save, "transforms.npy"),
             transforms[-1].matrix().detach().cpu().numpy(),
        )
        # np.save(
        #     os.path.join(path_save, "transforms_gt.npy"),
        #     data["transforms_gt"].detach().cpu().numpy(),
        # )
        #
        np.save(
            os.path.join(path_save, "positions.npy"),
             data["positions"].detach().cpu().numpy(),
        )

        save_volume(
                os.path.join(path_save, "2D_aligned.nii.gz"), volume_pred, data["resolution_recon"]
                )

