import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Mapping

def rotate90(img):
    """rotate batch of images by 90 degrees"""
    return torch.rot90(img, 3, (2, 3))
def rotate180(img):
    """rotate batch of images by 180 degrees"""
    return torch.rot90(img, 2, (2, 3))
def rotate270(img):
    """rotate batch of images by 270 degrees"""
    return torch.rot90(img, 1, (2, 3))

def hflip(img):
    """flip batch of images horizontally"""
    return img.flip(3)
def vflip(img):
    """flip batch of images vertically"""
    return img.flip(2)

class HorizontalFlip(object):
    """Flip images horizontally (left->right)"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return hflip(img=image)

    def apply_deaug_mask(self, mask,):
        return hflip(img=mask)

    def apply_deaug_label(self, label,):
        return hflip(img=label)
class VerticalFlip(object):
    """Flip images vertically (up->down)"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return vflip(img=image)

    def apply_deaug_mask(self, mask,):
        return vflip(img=mask)

    def apply_deaug_label(self, label,):
        return vflip(img=label)

class Rotate_0(object):
    """Rotate images 0 degrees"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return image

    def apply_deaug_mask(self, mask,):
        return mask

    def apply_deaug_label(self, label,):
        return label
    
class Rotate_90(object):
    """Rotate images 90 degrees"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return rotate90(img=image)

    def apply_deaug_mask(self, mask,):
        return rotate270(img=mask)

    def apply_deaug_label(self, label,):
        return rotate270(img=label)
    
class Rotate_180(object):
    """Rotate images 180 degrees"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return rotate180(img=image)

    def apply_deaug_mask(self, mask,):
        return rotate180(img=mask)

    def apply_deaug_label(self, label,):
        return rotate180(img=label)
    
class Rotate_270(object):
    """Rotate images 270 degrees"""
    def __init__(self) -> None:
        pass

    def apply_aug_image(self, image,):
        return rotate270(img=image)

    def apply_deaug_mask(self, mask,):
        return rotate90(img=mask)

    def apply_deaug_label(self, label,):
        return rotate90(img=label)
    
class Merger:
    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = torch.max(self.output, x)
        elif self.type == 'min':
            self.output = torch.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result
    

all_transforms = [
    Rotate_0(), 
    Rotate_90(),
    Rotate_180(),
    Rotate_270(),
    HorizontalFlip(),
    VerticalFlip(),
]

class PseudoAug(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with pseudo label augmentation 
    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (list): list of augmentation tranform
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """
    def __init__(
        self,
        model: nn.Module,
        transforms: List = all_transforms,
        merge_mode: str = "mean",
    ):
        super().__init__()
        self.model = model.eval()
        self.transforms = transforms
        self.merge_mode = merge_mode
    
    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        batch = image.size(0)
        result_list = torch.Tensor().to(image.get_device())
        for b in range(batch):
            merger = Merger(type=self.merge_mode, n=len(self.transforms))
            img = image[b].unsqueeze(dim=0)
            for transformer in self.transforms:
                augmented_image = transformer.apply_aug_image(img)
                with torch.no_grad():
                    augmented_output = self.model(augmented_image, *args)
                deaugmented_output = transformer.apply_deaug_mask(augmented_output)
                merger.append(deaugmented_output)
            result_list = torch.cat((result_list, merger.result))
        return result_list
    
    def update_model(
        self, model: nn.Module,
    ):
        self.model = model.eval()
    
if __name__ == "__main__":
    test_aug = ['Rotate_0', 'Rotate_90', 'Rotate_180', 'Rotate_270', 'HorizontalFlip', 'VerticalFlip']
    test_ = np.arange(0,len(test_aug))
    print(test_)
    weight = np.ones((len(test_aug)))/len(test_aug)
    print(weight)

    a = np.random.choice(test_, size=4, replace=False, p=weight)
    print(a)