from .aug_pil_list import *
from .aug_pil_method import *

class RandAugment_best_2aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_2Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img
    
class RandAugment_new_best_4aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_new_best_4Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img
    
class RandAugment_best_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img
    
class RandAugment_best9aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_9Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img