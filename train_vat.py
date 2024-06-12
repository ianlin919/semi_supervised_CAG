## Basic
import os
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import ssl
from utils import *
from itertools import cycle
from datetime import datetime
## DeepLearning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import  lightning.pytorch as pl
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
## Model
from networks import net_factory
import segmentation_models_pytorch as smp
## Data & Augmentation
from dataset import ImageDataSet_Train, ImageDataSet_Valid, ImageDataSet_Semi1
from transforms import get_train_augmentation
from transforms.RandAugment import RandAugment_best_2aug
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=True, ema=False):
    # net = smp.Unet("efficientnet-b6", in_channels=1, classes=1)
    net = net_factory('unet',1,1)  # 'unet','enet','pnet'
    if norm:
        model = kaiming_normal_init_weight(net)
    else:
        model = net
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def save_model(fabric,model, best_target, current_target, save_dir, epoch, best_epoch):
    if current_target > best_target:
        model_path = os.path.join(save_dir,'best.pt')
        torch.save(model.state_dict(), model_path)
        best_target = current_target
        best_epoch = epoch
        return best_target, best_epoch
    else:
        return best_target, best_epoch

def train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch):
    train_loader_l, train_loader_u = train_loader
    model.train()
    su_loss = 0
    un_loss = 0
    iter_num = 0
    iter_num = (epoch * len(train_loader_u))
    
    with tqdm(total=len(train_loader_u.dataset), desc="train ", unit="img",
              bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
        for (imgs, labels), un_w in zip(cycle(train_loader_l), train_loader_u):
            imgs, labels = imgs.float(), labels.float()
            un_w = un_w.float()
            assert labels.ndim == 4
            assert labels.max() <= 1.0 and labels.min() >= 0
            optimizer.zero_grad()
            un_batch_size = un_w.shape[0]
            """
            1. VAT Loss
            """
            vat_loss = VATLoss()
            lds = vat_loss(model, un_w)
            """
            2. Supervised Loss
            """
            preds = model(imgs)
            loss_su = loss(preds, labels)
            su_loss = su_loss + (loss_su.detach().clone().item() / imgs.shape[0])
            """
            3. Loss Backward
            """
            cons_weight = get_current_consistency_weight(iter_num)
            cons_loss = cons_weight * lds
            un_loss = un_loss + (cons_loss.detach().clone().item() / un_batch_size)
            
            loss_ = loss_su + cons_loss
            fabric.backward(loss_)
            optimizer.step()
            
            iter_num = iter_num + 1
            pbar.update(un_batch_size)
            
    scheduler.step()
    train_history["train"]["loss"].append(su_loss + un_loss)
    return train_history

def valid(fabric, model, valid_loader, train_history):
    model.eval()
    v_SE = 0.0
    v_SP = 0.0
    v_PR = 0.0
    v_F1 = 0.0
    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="valid ", unit="img",
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            for imgs, labels in valid_loader:
                imgs, labels = imgs.float(), labels.float()
                preds = model(imgs)
                SP = get_specificity(preds, labels)
                SE = get_sensitivity(preds, labels)
                PR = get_precision(preds, labels)
                F1 = get_F1(preds, labels)
                v_SE = v_SE + SE
                v_SP = v_SP + SP
                v_PR = v_PR + PR
                v_F1 = v_F1 + F1
                pbar.update(imgs.shape[0]) 
    train_history["valid"]["SE"].append(v_SE / len(valid_loader))
    train_history["valid"]["SP"].append(v_SP / len(valid_loader))
    train_history["valid"]["PR"].append(v_PR / len(valid_loader))
    train_history["valid"]["F1"].append(v_F1 / len(valid_loader))
    logs = {'valid_dice_score': v_F1 / len(valid_loader),
            'valid_precision': v_PR / len(valid_loader),
            'valid_sensitivity': v_SE / len(valid_loader),
            'valid_specificity': v_SP / len(valid_loader),
            }
    fabric.log_dict(logs)
    return train_history, (v_F1 / len(valid_loader))

def test(fabric, model, valid_loader, save_dir):
    model_path = os.path.join(save_dir,'best.pt')
    model.load_state_dict(torch.load(model_path))
    from lightning.fabric import is_wrapped
    if not is_wrapped(model):
        model = fabric.setup(model)
    if not is_wrapped(valid_loader):
        valid_loader = fabric.setup_dataloaders(valid_loader)
    model.eval()
    v_SE = 0.0
    v_SP = 0.0
    v_PR = 0.0
    v_F1 = 0.0
    with torch.no_grad():
        with tqdm(total=len(valid_loader.dataset), desc="valid ", unit="img",
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            for imgs, labels in valid_loader:
                imgs, labels = imgs.float(), labels.float()
                preds = model(imgs)
                SP = get_specificity(preds, labels)
                SE = get_sensitivity(preds, labels)
                PR = get_precision(preds, labels)
                F1 = get_F1(preds, labels)
                v_SE = v_SE + SE
                v_SP = v_SP + SP
                v_PR = v_PR + PR
                v_F1 = v_F1 + F1
                pbar.update(imgs.shape[0]) 
    
    logs = {'valid_dice_score': v_F1 / len(valid_loader),
            'valid_precision': v_PR / len(valid_loader),
            'valid_sensitivity': v_SE / len(valid_loader),
            'valid_specificity': v_SP / len(valid_loader),
            }
    print(logs)

def main(args):
    moving_dot_product = torch.empty(1)
    limit = 3.0 ** (0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)
    train_history = createTrainHistory(["loss", "SE", "SP", "PR", "F1"])
    csv_logger = CSVLogger(root_dir=args.s_dir)
    fabric = Fabric(accelerator="gpu", 
                    devices=[args.g],
                    precision="16-mixed",
                    loggers=csv_logger)
    fabric.launch()
    """
    Setup Model, Optimizer, Scheduler, Loss
    Option: pytorch 2.0 compile model
    """
    model = create_model()
    print("=" * 50)
    print("model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    print("=" * 50)
    loss = DiceBCELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.e)
    model, optimizer = fabric.setup(model, optimizer)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet_Train(
        txt_path=args.t_txt_path,
        img_dir=args.i_dir,
        label_dir=args.l_dir,
        size=args.size,
        transform=get_train_augmentation())
    valid_dataset = ImageDataSet_Valid(
        txt_path=args.v_txt_path,
        img_dir=args.i_dir,
        label_dir=args.l_dir,
        size=args.size,
        sort=True)
    train_dataset_u = ImageDataSet_Semi1(
        un_txtPath=args.t_un_txt_path,
        img_dir=args.i_dir,
        size=args.size,
        transform=RandAugment_best_2aug(1, 20),
        transform_weak=weak_aug())
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bl,
        num_workers=args.c,
        shuffle=True,
        pin_memory=True,
        drop_last=True,)
    train_loader_u = DataLoader(
        dataset=train_dataset_u,
        batch_size=args.bu,
        shuffle=True,
        num_workers=args.c,
        pin_memory=True,
        drop_last=True,)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=args.c,
        shuffle=False,
        pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    train_loader_u = fabric.setup_dataloaders(train_loader_u)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    
    best_target = 0
    best_epoch = 0
    
    for epoch in range(args.e):
        train_history = train(fabric, model, [train_loader, train_loader_u], 
                              loss, optimizer, scheduler, train_history)
        train_history, current_target = valid(fabric, model, valid_loader, train_history)
        best_target, best_epoch = save_model(fabric, model, best_target, current_target, args.s_dir, epoch, best_epoch)
        print(f"Epoch {epoch + 1} "
              f"loss : {train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history['valid']['F1'][epoch]:2.5f}")
        print("Best F1/DSC {} on Epoch {}".format(str(best_target),str(best_epoch)))
    fabric.logger.save()
    saveDict(args.s_dir/'train_history.pickle', train_history)
    test(fabric, model, valid_loader, args.s_dir)
    return

if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument("--bl", "--batch_size_labeled", default=4, type=int)
    parser.add_argument("--bu", "--batch_size_un", default=10, type=int)
    parser.add_argument("--e", "--Epoch", default=300, type=int)
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=2, type=int)
    """
    CAG Dataset Labeled 500 and Unlabeled 8952
    Dataset split to train and valid : 400/100
    Image resize to (512,512)
    # """
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/cag/imgs"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/cag/labels"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/VAT/F2/20_500/best")) #s_add_augment pseudoaug_ada
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/cag/labeled_400_2.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/cag/unlabeled_all.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
    
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/cag/labeled_20_2.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/cag/un_500.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
    """
    STARE Dataset Labeled 20 and Unlabeled 377
    Dataset split to train and valid : 18/2
    Image resize to (704,704)
    """
    # parser.add_argument("--size", "--image_size", default=(704,704))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/stare/original"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/stare/label"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/STARE/VAT/18_338"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/stare/train3.txt")
    # # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled_clean.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/stare/val3.txt")
    """
    DCA1 Dataset Labeled 134
    Dataset split to train and valid : 100/34
    Image resize to (320,320)
    """
    parser.add_argument("--size", "--image_size", default=(320,320))
    parser.add_argument("--i_dir", "--img_dir", default=Path("./data/DCA1/img"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("./data/DCA1/gt"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/DCA1/10_90/VAT"))
    parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/DCA1/DCA1_train_l_10.txt") # DCA1_train_l_5 DCA1_train_l_10
    parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/DCA1/DCA1_train_u_90.txt") # DCA1_train_u_95 DCA1_train_u_90 
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/DCA1/DCA1_val_34.txt")
    
    parser.add_argument("--lr", "--learning_rate", default=1e-3)
    args = parser.parse_args()
    
    begin_time = datetime.now()
    main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)