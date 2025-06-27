import os
import yaml
import torch
import nibabel as nib
import numpy as np
from scipy import ndimage
from monai.transforms import ScaleIntensity


def load_nifti_tensor(path, device):
    vol = nib.load(path + '.nii.gz').get_fdata()
    vol = torch.from_numpy(vol[None, None, :, :, :])
    vol = torch.flip(vol.permute(0, 1, 4, 3, 2), (-1, -2))
    return vol.contiguous().to(dtype=torch.float32, device=device)


class EchoDataset:
    def __init__(self, is_test, is_val, cfg):
        self.is_test = is_test
        self.is_val = is_val
        self.device = torch.device(cfg["device"])
        self.data_dir = cfg["data_dir"]
        self.test_set = cfg["test_set"]
        self.val_set = cfg["val_set"]
        self.mask_heart_path = cfg["mask_heart"]
        self.mask_thorax_path = cfg["mask_thorax"]
        self.atlas_path = cfg["atlas"]
        self.files = self.get_file()
        self.idx = 0
        self.res = cfg["resolution_training"]
        self.rescale = ScaleIntensity(0, 1)

    def get_file(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        file_list_path = os.path.join(parent_dir, 'config', 'dataset.yaml')

        with open(file_list_path, 'r') as f:
            file_config = yaml.safe_load(f)


        files = []
        for i, entry in enumerate(file_config['data_files']):
            if self.is_test and i not in self.test_set:
                continue
            if self.is_val and i not in self.val_set:
                continue
            if not self.is_test and i in self.test_set:
                continue
            if not self.is_val and i in self.val_set:
                continue

            # Append paired data (if available, for testing)
            if 'image_2d' in entry:
                files.append([
                    entry['folder'],
                    entry['volume'],
                    entry['image_2d'],
                    entry['PE_2D'],
                ])
            else:
                files.append([entry['folder'],
                              entry['volume']])

        return files

    def __len__(self):
        return len(self.files)

    def get_data(self):

        data = {"resolution": self.res}

        if len(self.files[self.idx])>2:
            folder, stic_fn, fn_2d, PE = self.files[self.idx]
            # Load 2D file and PE
            img_2d = load_nifti_tensor(os.path.join(self.data_dir, folder, fn_2d), self.device)
            img_2d = self.rescale(img_2d)
            data['PE_2D'] = PE
            data['img_2d'] = img_2d

        else:
            folder, stic_fn = self.files[self.idx][0]

        # Testing on real 2D data
        if self.is_test:
            self.idx += 1
            data['test_transform'] = True
        else:
            self.idx = np.random.choice(len(self.files))

        volume = load_nifti_tensor(os.path.join(self.data_dir, folder, stic_fn), self.device)
        volume = self.rescale(volume)


        mask_heart = load_nifti_tensor(self.mask_heart_path, self.device) if self.mask_heart_path else None
        mask_thorax = load_nifti_tensor(self.mask_thorax_path, self.device) if self.mask_thorax_path else None

        available_masks = []
        if mask_heart is not None:
            available_masks.append(("heart", mask_heart))
            data['mask_heart'] = mask_heart
        if mask_thorax is not None:
            available_masks.append(("thorax", mask_thorax))

        # Option with only one mask available
        if available_masks and not self.is_test:
            # Add "none" to simulate no masking
            choice, mask_tensor = np.random.choice(available_masks + [("none", None)])

            if choice == "heart":
                mask = (mask_tensor > 0).float()
                if np.random.rand() > 0.5:
                    dilated = ndimage.binary_dilation(mask_tensor.cpu().numpy(), iterations=5)
                    mask = torch.from_numpy(dilated).to(dtype=torch.float32, device=self.device)
                volume *= mask
            elif choice == "thorax":
                mask = (mask_tensor > 0).float()
                volume *= mask
            # If "none", skip masking

        if self.atlas_path:
            # Load atlas (if available)
            vol_atlas = load_nifti_tensor(self.atlas_path, self.device)
            vol_atlas = self.rescale(vol_atlas)
            data["atlas"] = vol_atlas.clone()

        data["volume"] = volume
        return data
