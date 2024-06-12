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
from dataset import ImageDataSet_Train, ImageDataSet_Valid
from transforms import get_train_augmentation, get_train_simple
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=True):
    net = net_factory('ske',1,1) # fr_unet unext
    # 'unet','enet','pnet' efficient_unet
    if norm:
        model = kaiming_normal_init_weight(net)
    else:
        model = net
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

def train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, args, epoch):
    model.train()
    su_loss = 0
    iter_num = (epoch * len(train_loader))
    max_iterations = (args.e * len(train_loader))
    with tqdm(total=len(train_loader.dataset), desc="train ", unit="img",
              bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
        for imgs, labels in train_loader:
            imgs, labels = imgs.float(), labels.float()
            label_128 = F.interpolate(labels.detach().clone(), (128, 128))
            label_64 = F.interpolate(labels.detach().clone(), (64, 64))
            label_32 = F.interpolate(labels.detach().clone(), (32, 32))
            
            assert labels.ndim == 4
            assert labels.max() <= 1.0 and labels.min() >= 0
            optimizer.zero_grad()
            preds, aux_128, aux_64, aux_32 = model(imgs)
            loss_su = loss(preds, labels)
            loss_128 = loss(aux_128, label_128)
            loss_64 = loss(aux_64, label_64)
            loss_32 = loss(aux_32, label_32)
            loss_ = loss_su*0.5 + loss_128*0.3 + loss_64*0.2 + loss_32*0.1
            fabric.backward(loss_)
            optimizer.step()
            su_loss = su_loss + (loss_su.item() / imgs.shape[0])
            pbar.update(imgs.shape[0])

    scheduler.step()
    train_history["train"]["loss"].append(su_loss)
    return train_history

def valid(fabric, model, valid_loader, train_history):
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
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=args.e, eta_min=1e-5)
    model, optimizer = fabric.setup(model, optimizer)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet_Train(
        txt_path=args.t_txt_path,
        img_dir=args.i_dir,
        label_dir=args.l_dir,
        size=args.size,
        transform=get_train_simple())
    valid_dataset = ImageDataSet_Valid(
        txt_path=args.v_txt_path,
        img_dir=args.i_dir,
        label_dir=args.l_dir,
        size=args.size,
        sort=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bl,
        num_workers=args.c,
        shuffle=True,
        pin_memory=True,
        drop_last=True,)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=args.c,
        shuffle=False,
        pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    
    best_target = 0
    best_epoch = 0
    
    for epoch in range(args.e):
        train_history = train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, args, epoch)
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
    from test import test
    test(fabric, model, valid_loader, args.s_dir)
    return

if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument("--bl", "--batch_size_labeled", default=16, type=int)
    parser.add_argument("--e", "--Epoch", default=300, type=int)
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=0, type=int)
    """
    SkelNetOn2019 Dataset Labeled
    Dataset split to train and valid : 90%/10%
    Image resize to (512,512)
    """
    parser.add_argument("--size", "--image_size", default=(256,256))
    parser.add_argument("--i_dir", "--img_dir", default=Path("./data/SkelNetOn2019/imgs"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("./data/SkelNetOn2019/labels"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/SkelNetOn2019/unet_scse_ham_wrs"))
    parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/SkelNetOn2019/labeled_seed_42.txt")
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/SkelNetOn2019/val_seed_42.txt")
 
    parser.add_argument("--lr", "--learning_rate", default=1e-3)
    args = parser.parse_args()
    
    begin_time = datetime.now()
    main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)