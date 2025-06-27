from data.dataset import EchoDataset
from test import *
from transform import mat_update_resolution
from slice_acquisition import slice_acquisition


def compute_similarity_metrics(mat, volume, data, fname):
    """

    Args:
        mat: transformation matrix (predicted or GT)
        volume: string ("volume") or ("atlas) representing the data key from where to acquire slices using mat
        data: data dict (from data/scan)
        fname: path for saving csv

    Saves metrics to a csv on specified path

    """
    df = pd.DataFrame()
    slices_pred = slice_acquisition(
                                    mat,
                                    data[volume],
                                    None,
                                    None,
                                    data["psf_acq"],
                                    data["slice_shape"],
                                    data["resolution_slice"] / data["resolution_recon"],
                                    False,
                                    False, )

    if "mask_heart" in data:
        slices_mask = slice_acquisition(
            mat,
            data["mask_heart"],
            None,
            None,
            data["psf_acq"],
            data["slice_shape"],
            data["resolution_slice"] / data["resolution_recon"],
            False,
            False, )

        slices_pred *= (slices_mask > 0).float()

    metrics_vol_pred = process_slices(slices_pred, data['seqs'], mask=True)

    df['NMI (Vol Pred, 2D In)'] = metrics_vol_pred[0]
    df['NCC (Vol Pred, 2D In)'] = metrics_vol_pred[1]
    df['SSIM (Vol Pred, 2D In)'] = metrics_vol_pred[2]
    df['PSNR (Vol Pred, 2D In)'] = metrics_vol_pred[3]

    if i > 0:
        df.to_csv(fname, mode='a', header=False)
    else:
        df.to_csv(fname)

    return 0

if __name__ == "__main__":
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="path to the yaml config file", required=True, type=str
    )
    parser.add_argument("--transforms", help="transforms path", required=True, type=str)
    parser.add_argument("--output", help="output path", required=True, type=str)
    args = parser.parse_args()

    cfg = get_config(args.config)
    # mkdir
    os.makedirs(args.output, exist_ok=True)
    # model
    device = torch.device(cfg["model"]["device"])

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

        transforms = RigidTransform(
            torch.from_numpy(np.load(os.path.join(args.transforms,str(i),
                                                  "transforms.npy"))).cuda())
        mat = transforms.matrix()

        mat = mat_update_resolution(mat,
                                    cfg["scanner"]["resolution_transform"],
                                    cfg["scanner"]["resolution_recon"])

        if "volume" in data:
            fname = os.path.join(args.output, "metrics_volume.csv")
            compute_similarity_metrics(mat, "volume", data, fname)

        if "atlas" in data:
            fname = os.path.join(args.output, "metrics_atlas.csv")
            compute_similarity_metrics(mat, "atlas", data, fname)


        # Generate high-res alignment
        aligned_2d = slice_acquisition_adjoint(
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

        # Save high-res aligned data
        save_volume(
            os.path.join(path_save, "aligned_2d.nii.gz"), aligned_2d, data["resolution_recon"]
        )

