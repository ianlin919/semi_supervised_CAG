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
from networks.utnet import UTNet
import segmentation_models_pytorch as smp
## Data & Augmentation
from dataset import ImageDataSet_Train, ImageDataSet_Valid,ImageDataSet_supervised_aug
from transforms import get_train_augmentation, get_train_simple
from transforms.RandAugment import RandAugment_best_2aug_with_ori_img
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=False, ema=False, freez=False, backbone=False):
    net = UTNet(1, 64, 1, reduce_size=16, block_list='1234', num_blocks=[1,1,1,1],
                num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, maxpool=True)
    # efficientnet-b6 tu-efficientnetv2_l tu-tf_efficientnetv2_l mobileone_s1
    # timm-resnest101e
    # net = net_factory('unet',1,1) # fr_unet unext
    # 'unet','enet','pnet' efficient_unet
    if backbone:
        pass
        # net.load_state_dict(torch.load('./logs/CAG/supervised/F2/400/efficientnet_b3_none_norm/best.pt'))
        # net.initialize(net.decoder)
    if freez:
        for param in net.encoder.parameters():
            param.requires_grad=False
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

def train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, args, epoch):
    model.train()
    su_loss = 0
    iter_num = (epoch * len(train_loader))
    max_iterations = (args.e * len(train_loader))
    with tqdm(total=len(train_loader.dataset), desc="train ", unit="img",
              bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
        for imgs, labels in train_loader:
            imgs, labels = imgs.float(), labels.float()
            assert labels.ndim == 4
            assert labels.max() <= 1.0 and labels.min() >= 0
            optimizer.zero_grad()
            preds = model(imgs)
            loss_su = loss(preds, labels)
            fabric.backward(loss_su)
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

# def test(fabric, model, valid_loader, save_dir):
#     model_path = os.path.join(save_dir,'best.pt')
#     model.load_state_dict(torch.load(model_path))
#     from lightning.fabric import is_wrapped
#     if not is_wrapped(model):
#         model = fabric.setup(model)
#     if not is_wrapped(valid_loader):
#         valid_loader = fabric.setup_dataloaders(valid_loader)
#     model.eval()
#     v_SE = 0.0
#     v_SP = 0.0
#     v_PR = 0.0
#     v_F1 = 0.0
#     with torch.no_grad():
#         with tqdm(total=len(valid_loader.dataset), desc="test ", unit="img",
#                     bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
#             for imgs, labels in valid_loader:
#                 imgs, labels = imgs.float(), labels.float()
#                 preds = model(imgs)
#                 SP = get_specificity(preds, labels)
#                 SE = get_sensitivity(preds, labels)
#                 PR = get_precision(preds, labels)
#                 F1 = get_F1(preds, labels)
#                 v_SE = v_SE + SE
#                 v_SP = v_SP + SP
#                 v_PR = v_PR + PR
#                 v_F1 = v_F1 + F1
#                 pbar.update(imgs.shape[0]) 
    
#     logs = {'valid_dice_score': v_F1 / len(valid_loader),
#             'valid_precision': v_PR / len(valid_loader),
#             'valid_sensitivity': v_SE / len(valid_loader),
#             'valid_specificity': v_SP / len(valid_loader),
#             }
#     print(logs)
    

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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=args.e)
    # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=600)
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
                    train_dataset,
                   batch_size=args.bl,
                   shuffle=True,
                   num_workers=args.c,
                   pin_memory=True,
                   drop_last=True)
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
    parser.add_argument("--bl", "--batch_size_labeled", default=10, type=int)
    parser.add_argument("--e", "--Epoch", default=200, type=int)
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=0, type=int)
    """
    FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation
    https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1
    FIVES Dataset Labeled 800, train/test 600/200(official split)
    Image resize from (2048, 2048) to (512,512)
    """
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/FIVES/img"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/FIVES/gt"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/FIVES/180/supervised")) # 30 60 120 180 600
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/FIVES/FIVES_train_l_180.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/FIVES/FIVES_valid_200.txt")
    """
    DCA1 Dataset Labeled 134
    Dataset split to train and valid : 100/34
    Image resize to (320,320)
    """
    # parser.add_argument("--size", "--image_size", default=(320,320))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/DCA1/img"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/DCA1/gt"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/DCA1/30/sup_unet"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/DCA1/DCA1_train_l_30.txt") # DCA1_train_l_5 DCA1_train_l_10 DCA1_train_100
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/DCA1/DCA1_val_34.txt")
    """
    STARE Dataset Labeled 20 and Unlabeled 377
    Dataset split to train and valid : 18/2
    Image resize to (704,704)
    """
    # parser.add_argument("--size", "--image_size", default=(704,704))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/stare/original"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/stare/label"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/STARE/F6/18/de_dcgcn_ee_simaug2"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/stare/train6.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/stare/val6.txt")
    """
    CAG Dataset Labeled 500 and Unlabeled 8952
    Dataset split to train and valid : 400/100
    Image resize to (512,512)
    """
    parser.add_argument("--size", "--image_size", default=(512, 512))
    parser.add_argument("--i_dir", "--img_dir", default=Path("/home/ryan0208/python/semi/data/cag/imgs2"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("/home/ryan0208/python/semi/data/cag/labels2"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/supervised/F2/400/UTNet_64patches_bs10"))
    parser.add_argument("--t_txt_path", "--train_txt_path",
                        default="/home/ryan0208/python/semi/data/cag/labeled_400_2.txt")
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="/home/ryan0208/python/semi/data/cag/valid_2.txt")
    
    # # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/cag/imgs"))
    # # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/cag/labels"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/F2/50/supervised/unet"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/cag/labeled_400_2.txt") # labeled_771_2 labeled_629_2
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
    
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/cag/labeled_50_2.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
 
    parser.add_argument("--lr", "--learning_rate", default=1e-2)
    args = parser.parse_args()
    
    begin_time = datetime.now()
    main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)