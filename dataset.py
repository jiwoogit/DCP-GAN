from io import BytesIO
import os

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

import lmdb

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class FFHQ_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''
    
    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.images_list = sorted([os.path.join(image_folder, image) for image in images_list])
        self.transform = transform
    
    def __getitem__(self, index):
        img_id = self.images_list[index]
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.images_list)

class church_dataset(Dataset):

    def __init__(self, folder, transform=None):
        self.transform = transform
        EXTS = ['jpg', 'jpeg', 'png']
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
