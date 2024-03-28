import torch
from torch.utils.data import Dataset

# fastmri
import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data

# Preparing and preprocessing fastMRI dataset
class fastMRIDataset(Dataset):
    def __init__(self, isval=False):
        self.isval = isval
        if not isval:
            self.data_path = '/home/haneol.kijm/Works/data/fastmri/knee_singlecoil_train/singlecoil_train/' # Adjust training data path here
        else:
            self.data_path = '/home/haneol.kijm/Works/data/fastmri/knee_singlecoil_val/singlecoil_val/' # Adjust validation data path here

        self.data = mri_data.SliceDataset(
            root=self.data_path,
            transform=self.data_transform,
            challenge='singlecoil',
            use_dataset_cache=True,
            )

        self.mask_func = subsample.RandomMaskFunc(
            center_fractions=[0.08],
            accelerations=[4],
            )
            
    def data_transform(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.isval:
            seed = tuple(map(ord, filename))
        else:
            seed = None     
        kspace = transforms.to_tensor(kspace)
        masked_kspace, _, _ = transforms.apply_mask(kspace, self.mask_func, seed)        
        
        target = transforms.to_tensor(target)
        zero_fill = fastmri.ifft2c(masked_kspace)
        zero_fill = transforms.complex_center_crop(zero_fill, target.shape)   
        x = fastmri.complex_abs(zero_fill)
 
        x = x.unsqueeze(0)
        target = target.unsqueeze(0)

        return (x, target, data_attributes['max'])

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        return data