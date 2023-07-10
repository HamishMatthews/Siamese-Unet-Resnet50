from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
import numpy as np
import h5py

class FireDataset(Dataset):
    """
    Custom dataset class for fire data.
    
    Args:
        hdf5_files (list): List of HDF5 file paths containing fire data.
        folds (list, optional): List of fold indices to include in the dataset.
    """
    def __init__(self, hdf5_files, folds=None):
        self.hdf5_files = hdf5_files
        self.folds = folds
        self.lengths = []
        self.total_length = 0
        
        # Calculate the length of each hdf5 file
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as f:
                if self.folds is None:
                    length = len(f)
                else:
                    length = len([uuid for uuid, values in f.items() if "fold" in values.attrs and values.attrs["fold"] in folds and "pre_fire" in values])
                self.lengths.append(length)
                self.total_length += length
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        cumsum_lengths = np.cumsum([0] + self.lengths)
        file_idx = np.searchsorted(cumsum_lengths, idx, side='right') - 1
        idx_within_file = idx - cumsum_lengths[file_idx]
        hdf5_file = self.hdf5_files[file_idx]

        with h5py.File(hdf5_file, "r") as f:
            if self.folds is None:
                uuids = list(f.keys())
            else:
                uuids = [uuid for uuid, values in f.items() if "fold" in values.attrs and values.attrs["fold"] in self.folds and "pre_fire" in values]
            selected_uuid = uuids[idx_within_file]
            values = f[selected_uuid]

            post_img = values["post_fire"][...]
            pre_img = values["pre_fire"][...]
            mask = values["mask"][...]


        if mask.shape[0] == 1 and len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
            
        # Check the data type of the mask and convert if necessary
        if mask.dtype != np.uint16:
            print('changing data type')
            mask = mask.astype(np.uint16)
    
        # Add an extra channel of zeros for missing band 10 (cirrus clouds)
        zeros = np.zeros_like(post_img[..., 0])
        # Add the extra channel to the end of the image
        post_img = np.insert(post_img, 9, zeros, axis=-1)   
        pre_img = np.insert(pre_img, 9, zeros, axis=-1)

        # Normalize the images
        post_img = post_img.astype(np.float32) / 65535.0  # Normalize to range [0, 1]
        pre_img = pre_img.astype(np.float32) / 65535.0
        mask = mask.astype(np.float32)

        # Convert to PyTorch tensors and move channel dimension to match PyTorch's format
        post_img = from_numpy(post_img).permute(2, 0, 1)
        pre_img = from_numpy(pre_img).permute(2, 0, 1)
        mask = from_numpy(mask).permute(2, 0, 1)

        return pre_img, post_img, mask
