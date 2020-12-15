import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class data_4_prediction(Dataset):
    def __init__(self, data_patches):
        self.data = np.transpose(data_patches, (0,3,1,2))
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx,:,:,:]
        sample = {'data': data}
        return sample



class data_set(Dataset):
    """
    pytorch iterative data loader costomized to lcz data
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.loadData()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx,:,:,:]
        label = self.label[idx,:]
        sample = {'data': data, 'label': label}

        return sample

    def loadData(self):
        f = h5py.File(self.data_dir)
        self.data = np.array(f['dat'])
        self.data = np.transpose(self.data, (0,3,1,2))

        self.label = np.array(f['lab'])
        self.label[self.label == 4] = 0

        f.close()

