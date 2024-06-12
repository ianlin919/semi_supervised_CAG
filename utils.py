import math
from typing import Any
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import pickle

def saveDict(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def createTrainHistory(keywords):
    history = {"train": {}, "valid": {}}
    for words in keywords:
        history["train"][words] = list()
        history["valid"][words] = list()
    return history

import torch

SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
	# You can comment out this line if you are passing tensors of equal shape
	# But if you are passing output from UNet or something it will most probably
	# be with the BATCH x 1 x H x W shape
	outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

	intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
	union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

	iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

	thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

	return thresholded

def get_sensitivity(SR, GT, threshold=0.5):
	# Sensitivity == Recall
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TP : True Positive
	# FN : False Negative
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FN = ((SR == 0).float() + (GT == 1).float()) == 2
    
	SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

	return SE


def get_specificity(SR, GT, threshold=0.5):
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TN : True Negative
	# FP : False Positive
	TN = ((SR == 0).float() + (GT == 0).float()) == 2
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

	return SP


def get_precision(SR, GT, threshold=0.5):
	SR = torch.sigmoid(SR)
	SR = SR > threshold
	GT = GT == torch.max(GT)

	# TP : True Positive
	# FP : False Positive
	TP = ((SR == 1).float() + (GT == 1).float()) == 2
	FP = ((SR == 1).float() + (GT == 0).float()) == 2

	PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

	return PC


def get_F1(SR, GT, threshold=0.5):
	# Sensitivity == Recall
	SE = get_sensitivity(SR, GT, threshold=threshold)
	PC = get_precision(SR, GT, threshold=threshold)

	F1 = 2 * SE * PC / (SE + PC + 1e-6)

	return F1

def get_confusion_matrix(SR, GT, threshold=0.5):
    tp, fp, fn, tn = 0, 0, 0, 0
    
    SR = torch.sigmoid(SR)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).float() + (GT == 1).float()) == 2
    FP = ((SR == 1).float() + (GT == 0).float()) == 2
    FN = ((SR == 0).float() + (GT == 1).float()) == 2
    TN = ((SR == 0).float() + (GT == 0).float()) == 2
    
    tp = float(torch.sum(TP))
    fp = float(torch.sum(FP))
    fn = float(torch.sum(FN))
    tn = float(torch.sum(TN))
    
    return tp, fp, fn, tn

def get_metrics(SR, GT, threshold=0.5):
    smooth = 1e-6
    f1, precision, recall, specificity = 0.0, 0.0, 0.0, 0.0
    dice, iou, acc = 0.0, 0.0, 0.0
    tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
    
    tp, fp, fn, tn = get_confusion_matrix(SR, GT, threshold)
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    dice = (2 * tp) / ( (2 * tp) + fp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    iou = tp / (tp + fn + fp + smooth)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return f1, dice, precision, recall, specificity, iou, acc

from medpy_metrics import dc, asd, hd95
def get_metrics_medpy(SR, GT, threshold=0.5):
    SR = torch.sigmoid(SR)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    prediction = SR.cpu().detach().numpy()
    groundtruth = GT.cpu().detach().numpy()
    dice_ = dc(prediction, groundtruth)
    asd_ = asd(prediction, groundtruth)
    hd95_ = hd95(prediction, groundtruth)
    return dice_, hd95_, asd_


from sklearn.metrics import roc_auc_score
def get_metrics_auc(SR, GT, threshold=0.5):
    SR = torch.sigmoid(SR).cpu().detach().numpy().flatten()
    GT = GT == torch.max(GT)
    GT = GT.cpu().detach().numpy().flatten()
    auc = roc_auc_score(GT, SR)
    return auc

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    #Get cosine scheduler (LambdaLR).
    #if warmup is needed, set num_warmup_steps (int) > 0.
    def _lr_lambda(current_step):
        
        #_lr_lambda returns a multiplicative factor given an interger parameter epochs.
        #Decaying criteria: last_epoch
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))
    return model, ema_model

def get_current_consistency_weight(epoch):
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return (1.0 * sigmoid_rampup(epoch, 600))

def kaiming_normal_init_weight(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return model
    
    
import contextlib
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=0.1, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
    
    
def get_r_adv_t(model, x , it=1, xi=1e-1, eps=10.0):
    # stop bn
    model.eval()
    
    x_detached = x.detach()
    with torch.no_grad():
        # get the ensemble results from teacher
        pred = F.softmax(model(x), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        pred_hat = model(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        model.zero_grad()

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    r_adv = d * eps
    # 
    pred_hat = model(x + r_adv)
    logp_hat = F.log_softmax(pred_hat, dim=1)
    lds = F.kl_div(logp_hat, pred, reduction='batchmean')
    # 
    model.train()
    # return r_adv
    return lds

def get_r_adv_s(model_t, model_s, x , it=1, xi=1e-1, eps=10.0):
    # stop bn
    model_t.eval()
    model_s.eval()
    
    x_detached = x.detach()
    with torch.no_grad():
        # get the ensemble results from teacher
        pred = F.softmax(((model_t(x) + model_s(x)) / 2.0), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        pred_hat = (model_t(x_detached + xi * d) + model_s(x_detached + xi * d)) / 2.0
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        model_t.zero_grad()
        model_s.zero_grad()

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    r_adv = d * eps
    
    logp_hat = F.log_softmax(model_t(F.dropout(x, 0.2) + r_adv), dim=1)
    lds_t = F.kl_div(logp_hat, pred, reduction='batchmean')
    # 
    logp_hat = F.log_softmax(model_s(F.dropout(x, 0.3) + r_adv), dim=1)
    lds_s = F.kl_div(logp_hat, pred, reduction='batchmean')
    # 
    model_t.train()
    model_s.train()
    return lds_t, lds_s

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torch.nn.modules.loss import _Loss

class PolyBCELoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 5.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        
        # # target is in the one-hot format, convert to BH[WD] format to calculat ce loss
        self.bce_loss = self.bce(input, target)
        # v3
        # pt = torch.mean((1-target) * torch.sigmoid(input),dim=-1)
        # v2
        pt = torch.mean(target*torch.sigmoid(input),dim=-1)
        # v1
        # pt_ = torch.sigmoid(input) 
        # pt = torch.where(target == 1, pt_, 1-pt_)
        
        # v3
        # poly_loss_ = []
        # for i in range(1, int(self.epsilon)+1):
        #     term  = torch.pow((1 - pt),i)/i
        #     poly_loss_.append(term)
        # poly_loss = self.bce_loss + sum(poly_loss_)/self.epsilon
        # v1, v2
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)
    
from monai.losses.dice import GeneralizedDiceLoss, DiceLoss
class PolyDiceLoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.dice = GeneralizedDiceLoss(sigmoid=True)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        
        # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.dice_loss = self.dice(input, target)
        # v2
        pt = torch.mean(target*torch.sigmoid(input),dim=-1)
        # v1
        # pt_ = torch.sigmoid(input) 
        # pt = torch.where(target == 1, pt_, 1-pt_)
        poly_loss = self.dice_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)

from monai.losses import FocalLoss
class PolyFocalLoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 gamma: float = 2.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.gamma = gamma
        self.focal = FocalLoss(gamma=self.gamma)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        p = torch.sigmoid(input)
        pt = (target * p) + ((1 - target) * (1 - p))
        
        self.focal_loss = self.focal(p, target)
        poly_loss = self.focal_loss + self.epsilon * torch.pow((1 - pt), self.gamma+1)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)
    
class DiceBCELoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        # self.bce = PolyBCELoss()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        
        # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.dice_loss = self.dice(input, target)
        self.bce_loss = self.bce(input, target)
        loss = (0.5 * self.bce_loss) + (self.dice_loss * 1)

        if self.reduction == 'mean':
            polyl = torch.mean(loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)

import random
import torchvision.transforms.functional as TF
class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
import torchvision.transforms as T
def gauss_noise_tensor(img):
    if random.random() < 0.5:
        out = img
    else:
        noise = torch.clamp(torch.randn_like(img) * 0.1, -0.2, 0.2)
        out = img + noise
    return out

def weak2strong():
    return T.Compose(transforms=[
        T.RandomChoice(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                   T.RandomErasing(),
                                   T.RandomAutocontrast(),
                                #    T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                                #    T.RandomEqualize(),
                                #    T.RandomPosterize(bits=2)
                                   ],),
        # MyRotationTransform(angles=[-90, 0, 90]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # gauss_noise_tensor
        # T.Resize((512,512)),
        ])

def strong_aug():
    return T.Compose(transforms=[
        T.RandomChoice(transforms=[
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandomErasing(),
            T.RandomAutocontrast(),
            # T.ColorJitter(0.2, 0.2, 0.2, 0.2),
            ],),
    ]) 
   
def weak_aug():
    return T.Compose(transforms=[
        MyRotationTransform(angles=[-90, 0, 90]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])

def no_aug(img):
    return img

from PIL import Image
from pathlib import Path
import os
class visualize_pseudol_label(object):
    def __init__(self,
                 device,
                 threshold=0.5,
                 name_save='test_th',
                 name_img='CVAI-0107_LAD_RAO12_CRA16_32.png' ,
                 path_img='./data/cag/imgs', 
                 path_gt='./data/cag/labels', 
                 path_dir='./logs/CAG/F2/20_500',
                 ):
        self.teacher_result = {
            'img':[],
            'pred':[],
            'gt':[],
            'dsc':[],
            'dsc_th':[]}
        self.student_result = {
            'img':[],
            'pred':[],
            'gt':[],
            'dsc':[],
            'dsc_th':[]}
        self.device = device
        self.threshold = threshold
        self.name_img = name_img
        self.path_img = Path(path_img)
        self.path_gt = Path(path_gt)
        self.path_dir = Path(path_dir/name_save)
        
        self.img, self.gt = self.read_data()
        self.img_tensor, self.gt_tensor = self.convert_tensor()
        
    def __call__(self, model_t, model_s):
        input_data = self.img_tensor.detach()
        model_t.eval()
        model_s.eval()
        pred_t = model_t(input_data.detach().clone())
        pred_s = model_s(input_data.detach().clone())
        self.convert_np(pred_t, pred_s)
        self.get_metrics(pred_t, pred_s)
        model_t.train()
        model_s.train()
        del input_data, pred_t, pred_s
        
    def read_data(self,):
        img = Image.open(self.path_img/self.name_img).convert('L').resize((512,512), Image.BICUBIC)
        gt = Image.open(self.path_gt/self.name_img).convert('L').resize((512,512), Image.BICUBIC)
        img, gt = np.array(img), np.array(gt)
        assert img.shape == (512,512)
        assert gt.shape == (512,512)
        return img, gt
    
    def convert_tensor(self):
        img = np.expand_dims(self.img, (0,1))
        gt = np.expand_dims(self.gt, (0,1))
        img = torch.from_numpy(img).float()/255
        gt = torch.from_numpy(gt).float()
        assert img.shape == (1,1,512,512)
        assert gt.shape == (1,1,512,512)
        img = img.to(self.device)
        gt = gt.to(self.device)
        return img, gt
    
    def convert_np(self, result_t, result_s):
        result_t = torch.sigmoid(result_t)
        result_s = torch.sigmoid(result_s)
        result_t = result_t.detach().cpu().squeeze().numpy()
        result_s = result_s.detach().cpu().squeeze().numpy()
        
        assert result_t.shape == (512,512)
        assert result_s.shape == (512,512)
        self.teacher_result['img'].append(self.img)
        self.student_result['img'].append(self.img)
        self.teacher_result['gt'].append(self.gt)
        self.student_result['gt'].append(self.gt)
        self.teacher_result['pred'].append(result_t)
        self.student_result['pred'].append(result_s)
    
    def get_metrics(self, result_t, result_s):
        t_f1 = get_F1(result_t, self.gt_tensor)
        t_f1_th = get_F1(result_t, self.gt_tensor, threshold=self.threshold)
        s_f1 = get_F1(result_s, self.gt_tensor)
        s_f1_th = get_F1(result_s, self.gt_tensor, threshold=self.threshold)
        self.teacher_result['dsc'].append(t_f1)
        self.teacher_result['dsc_th'].append(t_f1_th)
        self.student_result['dsc'].append(s_f1)
        self.student_result['dsc_th'].append(s_f1_th)
        
    def save_result(self,):
        if os.path.exists(self.path_dir):
            pass
        else:
            os.makedirs(self.path_dir)
        saveDict(self.path_dir/'teacher.pickle', self.teacher_result)
        saveDict(self.path_dir/'student.pickle', self.student_result)
    