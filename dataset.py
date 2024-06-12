from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import cv2
from utils import weak_aug, no_aug

def loadTxt(filename):
    f = open(filename)
    context = list()
    for line in f:
        context.append(line.replace("\n", ""))
    return context

def default_loader(path):
    return Image.open(path)

def check_data_shape(img, size):
    h, w = size
    if len(img.shape) == 2:
        # (h, w) -> (h, w, 1)
        img = np.expand_dims(img, -1)
    elif len(img.shape) == 3:
        # (h, w, n) -> (h, w) -> (h, w, 1)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, -1)
    assert img.shape == (h, w, 1)
    img = img.astype(np.float32)
    return img

class ImageDataSet_Train(Dataset):
    def __init__(self, 
                 txt_path, 
                 img_dir, 
                 label_dir,
                 size,
                 transform,
                 loader=default_loader, 
                 sort=False):
        fileNames = [name for name in loadTxt(str(txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = size
        self.loader = loader
        self.transform = transform
        
    def preprocess(self, img, size, label=None, DA=False):
        img = img.resize(size, Image.BICUBIC)
        label = label.resize(size, Image.BICUBIC)
        img = np.array(img)
        label = np.array(label)
        if self.transform and DA:
                sample = self.transform(image=img, mask=label)
                img, label = sample['image'], sample['mask']
        img = check_data_shape(img, size)
        label = check_data_shape(label, size)
        # normalize
        if img.max() > 1.0:
            img = img / 255.0
        if label.max() > 1.0:
            label = label // 255.0
        # (h, w, 1) -> (1, h, w)
        img, label = img.transpose((2, 0, 1)), label.transpose((2, 0, 1))
        return img, label

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))
        img, label = self.preprocess(img, self.size, label=label, DA=True)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_Valid(Dataset):
    def __init__(self,
                 txt_path,
                 img_dir,
                 label_dir,
                 size,
                 loader=default_loader,
                 sort=False):
        fileNames = [name for name in loadTxt(str(txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = size
        self.loader = loader

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BICUBIC)
        img = np.array(img)
        img = check_data_shape(img, size)

        if label:
            # normalize
            if img.max() > 1.0:
                img = img // 255.0
        else:
            # normalize
            if img.max() > 1.0:
                img = img / 255.0
        # (h, w, 1) -> (1, h, w)
        img = img.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))
        img = self.preprocess(img, self.size)
        label = self.preprocess(label, self.size, label=True)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_Semi1(Dataset):
    def __init__(self, 
                 un_txtPath,
                 img_dir,
                 size,
                 transform,
                 labeled_texPath=None,
                 transform_weak=no_aug,
                 loader=default_loader,
                 sort=False):
        if labeled_texPath is not None:
            fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_texPath))]
        else:
            fileNames = [name for name in loadTxt(str(un_txtPath))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.size = size
        self.loader = loader
        self.transform = transform
        self.transform_weak = transform_weak

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BICUBIC)
        if DA_s: # RandAugment
            img = self.transform(img)
        img = np.array(img)
        img = check_data_shape(img, size)
        # normalize
        if img.max() > 1.0:
            img = img / 255.0
        # (h, w, 1) -> (1, h, w)
        img = img.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        img = self.preprocess(img, self.size)
        img = torch.from_numpy(img).float()
        img = self.transform_weak(img)
        return img

    def __len__(self):
        return len(self.fileNames)

"""
Our 
"""

class ImageDataSet_Semi2(Dataset):
    def __init__(self, 
                 un_txtPath,
                 img_dir,
                 size,
                 transform,
                 labeled_texPath=None,
                 transform_weak=no_aug,
                 loader=default_loader,
                 sort=False):
        if labeled_texPath is not None:
            fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_texPath))]
        else:
            fileNames = [name for name in loadTxt(str(un_txtPath))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.size = size
        self.loader = loader
        self.transform = transform
        self.transform_weak = transform_weak

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BICUBIC)
        if DA_s: # RandAugment
            img = self.transform(img)
        img = np.array(img)
        img = check_data_shape(img, size)
        # normalize
        if img.max() > 1.0:
            img = img / 255.0
        # (h, w, 1) -> (1, h, w)
        img = img.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        img1 = self.preprocess(img, self.size)
        img2 = self.preprocess(img, self.size, DA_s=True)
        img1, img2 = torch.from_numpy(img1).float(), torch.from_numpy(img2).float()
        img1, img2 = self.transform_weak(torch.cat((img1, img2))).chunk(2)
        return img1, img2

    def __len__(self):
        return len(self.fileNames)

"""
UniMatch
"""

class ImageDataSet_Semi3(Dataset):
    def __init__(self, 
                 un_txtPath,
                 img_dir,
                 size,
                 transform,
                 transform_weak=no_aug,
                 loader=default_loader,
                 sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.size = size
        self.loader = loader
        self.transform = transform
        self.transform_weak = transform_weak
        
    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BICUBIC)
        if DA_s: # RandAugment
            img = self.transform(img)
        img = np.array(img)
        img = check_data_shape(img, size)
        # normalize
        if img.max() > 1.0:
            img = img / 255.0
        # (h, w, 1) -> (1, h, w)
        img = img.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        img1 = self.preprocess(img, self.size)
        img2 = self.preprocess(img, self.size, DA_s=True)
        img3 = self.preprocess(img, self.size, DA_s=True)
        img1 = torch.from_numpy(img1).float()
        img2, img3 = torch.from_numpy(img2).float(), torch.from_numpy(img3).float()
        img1, img2, img3 = self.transform_weak(torch.cat((img1, img2, img3))).chunk(3)
        return img1, img2, img3

    def __len__(self):
        return len(self.fileNames)

"""
ST & ST++
"""

class ImageDataSet1(Dataset):
    def __init__(self,
                 txt_path,
                 img_dir,
                 size,
                 loader=default_loader,
                 transform=False,
                 sort=False):
        fileNames = [name for name in loadTxt(str(txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.size = size
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BICUBIC)
        img = np.array(img)
        img = check_data_shape(img, size)
        if label:
            # normalize
            if img.max() > 1.0:
                img = img // 255.0
        else:
            # normalize
            if img.max() > 1.0:
                img = img / 255.0
        # (h, w, 1) -> (1, h, w)
        img = img.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        img = self.preprocess(img, self.size)
        return torch.from_numpy(img)

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet4(Dataset):
    def __init__(self, 
                 txt_path,
                 un_txt_path,
                 img_dir, 
                 label_dir,
                 pseudo_dir,
                 size,
                 transform,
                 transform_weak,
                 loader=default_loader, 
                 sort=False):

        self.fileNames_l = [name for name in loadTxt(str(txt_path))]
        self.fileNames_u = [name for name in loadTxt(str(un_txt_path))]
        fileNames = self.fileNames_l + self.fileNames_u
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.pseudo_dir = pseudo_dir
        self.size = size
        self.loader = loader
        self.transform = transform
        self.transform_weak = transform_weak
        
    def preprocess(self, img, size, label=None, DA=False):
        img = img.resize(size, Image.BICUBIC)
        label = label.resize(size, Image.BICUBIC)
        img = np.array(img)
        label = np.array(label)
        if self.transform and DA:
                sample = self.transform(image=img, mask=label)
                img, label = sample['image'], sample['mask']
        img = check_data_shape(img, size)
        label = check_data_shape(label, size)
        # normalize
        if img.max() > 1.0:
            img = img / 255.0
        if label.max() > 1.0:
            label = label // 255.0
        # (h, w, 1) -> (1, h, w)
        img, label = img.transpose((2, 0, 1)), label.transpose((2, 0, 1))
        return img, label

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        if fileName in self.fileNames_l:
            label = self.loader(str(self.label_dir / fileName))
        else:
            label = self.loader(str(self.pseudo_dir / fileName))
        img, label = self.preprocess(img, self.size, label=label, DA=True)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.fileNames)