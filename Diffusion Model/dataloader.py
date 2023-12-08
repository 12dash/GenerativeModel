import struct
import numpy as np

import torch
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, img_gzip, label_gzip, base_dir):
        self.img_gzip_path = base_dir + img_gzip
        self.label_gzip_path = base_dir + label_gzip
        self.imgs, self.labels = None, None
        self.load()
        
    def load(self):
        with open(self.img_gzip_path,'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            self.imgs = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))\
                     .reshape((size, nrows, ncols, 1))
            self.imgs = np.transpose(self.imgs, (0,3,1,2))

        with open(self.label_gzip_path,'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32)/127.5 - 1
        label = torch.tensor(self.labels[idx])
        return img, label