## Basic
import os
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import ssl
from utils import *
from itertools import cycle
## DeepLearning
from loss import GeneralizedDiceLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import  lightning.pytorch as pl
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
## Model
from networks import net_factory
import segmentation_models_pytorch as smp
## Data & Augmentation
from dataset import ImageDataSet_Train, ImageDataSet_Valid, ImageDataSet_semi2
from transforms import get_train_augmentation
from transforms.RandAugment import RandAugment_best_2aug
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=True, ema=False):
    net = smp.Unet("efficientnet-b6", in_channels=1, classes=1)
    # net = net_factory('unet',1,1)  # 'unet','enet','pnet'
    if norm:
        model = kaiming_normal_init_weight(net)
    else:
        model = net
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def save_model(fabric,model, best_target, current_target, save_dir, epoch, best_epoch, name_):
    if current_target > best_target:
        name = "{}_epoch_{}_{:.7s}.pt".format(name_, best_epoch, str(best_target))
        try:
            os.remove(os.path.join(save_dir,name))
        except:
            if epoch != 1:
                print('no file exist')
        name = "{}_epoch_{}_{:.7s}.pt".format(name_, epoch, str(current_target))
        model_path = os.path.join(save_dir,name)
        torch.save(model.state_dict(), model_path)
        best_target = current_target
        best_epoch = epoch
        return best_target, best_epoch
    else:
        return best_target, best_epoch

def train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch, args):
    model_t, model_s = model
    optimizer_t, optimizer_s = optimizer
    scheduler_t, scheduler_s = scheduler
    train_loader_l, train_loader_u = train_loader
    loss_l, loss_u = loss
    
    model_t.train()
    model_s.train()
    su_loss = 0
    un_loss = 0
    base_lr = args.lr
    iter_num = (epoch * len(train_loader_u))
    max_iterations = len(train_loader_u) * args.e
    
    # applier = weak2strong()
    applier_w = weak_aug()
    applier_s = strong_aug()
    
    with tqdm(total=len(train_loader_u.dataset), desc="train ", unit="img",
              bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
        for (imgs, labels), (un_w, un_s) in zip(cycle(train_loader_l), train_loader_u):
            imgs, labels = imgs.float(), labels.float()
            un_w, un_s = un_w.float(), un_s.float()
            assert labels.ndim == 4
            assert labels.max() <= 1.0 and labels.min() >= 0
            optimizer_t.zero_grad()
            optimizer_s.zero_grad()
            un_batch_size = un_s.shape[0]
            """
            1. Teacher Model Part
            """
            imgs_all = torch.cat((imgs, un_w, un_s))
            preds_all = model_t(imgs_all)
            preds, pred_w, pred_s = preds_all.split([imgs.shape[0], un_w.shape[0], un_s.shape[0]])
            del imgs_all, preds_all
            """
            1.1 Supervised Labeled data (imgs, labels) and preds
            """
            loss_su = loss_l(preds, labels)
            su_loss = su_loss + (loss_su.detach().clone().item() / imgs.shape[0])
            """
            1.2. pesudo label by unlabeled data weak : pred_weak
            confidence level is pred_weak * threshold : mask
            """
            cons_weight = get_current_consistency_weight(iter_num)
            threshold = 0.9 - (0.4 * cons_weight)
            mask = (torch.sigmoid(pred_w).ge(threshold) + torch.sigmoid(pred_w).le(1-threshold)).to(dtype=torch.float32).detach()
            
            cons_loss = loss_u((torch.sigmoid(pred_s) * mask), (torch.sigmoid(pred_w) * mask)) # test2 best
            un_loss = un_loss + (cons_loss.detach().clone().item() / un_batch_size)
            # ====================================
            t_loss = loss_su + cons_weight * cons_loss # + ccr_loss
            fabric.backward(t_loss)
            optimizer_t.step()
            # ====================================
            """
            2. Student Model Part
            """
            """
            2.1. Student VAT Loss
            """
            cons_loss_vat = cons_weight * get_r_adv_t(model_s, un_w.detach().clone())
            un_loss = un_loss + (cons_loss_vat.detach().clone().item() / un_batch_size)
            """
            2.2. Student labeled data
            """
            student_l = model_s(imgs)
            s_loss_l = loss_l(student_l, labels)
            """
            2.3. Student unlabeled data
            """
            pseudo_label = torch.sigmoid(pred_w.detach().clone()).ge(threshold).to(dtype=torch.float32)
            un_w_a, mask_a = applier_w(torch.cat((un_w.detach().clone(), pseudo_label))).chunk(2)
            un_w_a = applier_s(un_w_a)
            s_loss_u = loss_l(model_s(un_w_a), mask_a)
            # ====================================
            s_loss = s_loss_l + s_loss_u + cons_loss_vat
            fabric.backward(s_loss)
            optimizer_s.step()
            # ====================================
            iter_num = iter_num + 1
            pbar.update(un_batch_size)
            
    scheduler_t.step()
    scheduler_s.step()
        
    train_history["train"]["loss"].append(su_loss + un_loss)
    
    return train_history

def valid(fabric, model, valid_loader, epoch, train_history):
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
    model_t = create_model()
    model_s = create_model()
    # model_t = torch.compile(model_t)
    # model_s = torch.compile(model_s)
    print("=" * 50)
    print("teacher model parameters: {:.2f}M".format(sum(p.numel() for p in model_t.parameters())/1e6))
    print("student model parameters: {:.2f}M".format(sum(p.numel() for p in model_s.parameters())/1e6))
    print("=" * 50)
    loss = DiceBCELoss()
    loss_un = nn.MSELoss()
    optimizer_t = torch.optim.AdamW(params=model_t.parameters(), lr=args.lr)
    # scheduler_t = get_cosine_schedule_with_warmup(optimizer=optimizer_t,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=600)
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_t, T_max=args.e)
    optimizer_s = torch.optim.AdamW(params=model_s.parameters(), lr=args.lr)
    # scheduler_s = get_cosine_schedule_with_warmup(optimizer=optimizer_s,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=600)
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_s, T_max=args.e, eta_min=(args.lr*0.8))
    model_t, optimizer_t = fabric.setup(model_t, optimizer_t)
    model_s, optimizer_s = fabric.setup(model_s, optimizer_s)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet_train(txt_path=args.t_txt_path,
                                  img_dir=args.i_dir,
                                  label_dir=args.l_dir,
                                  size=args.size,
                                  transform=get_train_augmentation())
    valid_dataset = ImageDataSet_valid(txt_path=args.v_txt_path,
                                 img_dir=args.i_dir,
                                 label_dir=args.l_dir,
                                 size=args.size,
                                 sort=True)
    train_dataset_u = ImageDataSet4(labeled_txt_path=args.t_txt_path,
                                    un_txtPath=args.t_un_txt_path,
                                    img_dir=args.i_dir,
                                    size=args.size,
                                    transform=RandAugment_best_2aug(1, 20),) # RandAugment_18 RandAugment_best_2aug
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.bl,
                              num_workers=args.c,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,)
    train_loader_u = DataLoader(dataset=train_dataset_u,
                                batch_size=args.bu,
                                shuffle=True,
                                num_workers=args.c,
                                pin_memory=True,
                                drop_last=True,)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=args.c,
                              shuffle=False,
                              pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    train_loader_u = fabric.setup_dataloaders(train_loader_u)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    
    best_target_t = 0
    best_epoch_t = 0
    best_target_s = 0
    best_epoch_s = 0
    train_history_t = createTrainHistory(["loss", "SE", "SP", "PR", "F1"])
    for epoch in range(args.e):
        train_history = train(fabric, [model_t, model_s], 
                              [train_loader, train_loader_u], 
                              [loss, loss_un], 
                              [optimizer_t, optimizer_s], 
                              [scheduler_t, scheduler_s], train_history, epoch, args)
        train_history, current_target = valid(fabric, model_s, valid_loader, epoch, train_history)
        best_target_s, best_epoch_s = save_model(fabric, model_s, best_target_s, current_target, args.s_dir, epoch, best_epoch_s, name_='student')
        train_history_t, current_target = valid(fabric, model_t, valid_loader, epoch, train_history_t)
        best_target_t, best_epoch_t = save_model(fabric, model_t, best_target_t, current_target, args.s_dir, epoch, best_epoch_t, name_='teacher')
        print(f"Epoch {epoch + 1} "
              f"loss : {train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history['valid']['F1'][epoch]:2.5f}")
        print(f"Epoch {epoch + 1} "
              f"valid_PR : {train_history_t['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history_t['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history_t['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history_t['valid']['F1'][epoch]:2.5f}")
        print("Teacher Best F1/DSC {} on Epoch {}".format(str(best_target_t),str(best_epoch_t)))
        print("Student Best F1/DSC {} on Epoch {}".format(str(best_target_s),str(best_epoch_s)))
    fabric.logger.save()
    # test(fabric, model_s, valid_loader, args.s_dir)
    saveDict(args.s_dir/'train_history_s.pickle', train_history)
    saveDict(args.s_dir/'train_history_t.pickle', train_history_t)
    return

if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument("--bl", "--batch_size_labeled", default=10, type=int)
    parser.add_argument("--bu", "--batch_size_un", default=40, type=int)
    parser.add_argument("--e", "--Epoch", default=300, type=int)
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=0, type=int)
    """
    CAG Dataset Labeled 500 and Unlabeled 8952
    Dataset split to train and valid : 400/100
    Image resize to (512,512)
    """
    parser.add_argument("--size", "--image_size", default=(512,512))
    parser.add_argument("--i_dir", "--img_dir", default=Path("/mnt/workspace/CAG/imgs2"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("/mnt/workspace/CAG/labels2"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/F2/all/S4_best")) #s_add_augment pseudoaug_ada
    parser.add_argument("--t_txt_path", "--train_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/labeled_400_2.txt")
    parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/unlabeled_all.txt")
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/valid_2.txt")
    
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/labeled_20_2.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/un_500.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="/mnt/workspace/semi_supervised_CAG/data/cag/valid_2.txt")
    """
    STARE Dataset Labeled 20 and Unlabeled 377
    Dataset split to train and valid : 18/2
    Image resize to (704,704)
    """
    # parser.add_argument("--size", "--image_size", default=(704,704))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/stare/original"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/stare/label"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/STARE/F6/18_377/S4_best"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/stare/train6.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled.txt")
    # # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled_clean.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/stare/val6.txt")
    """
    DCA1 Dataset Labeled 134
    Dataset split to train and valid : 100/34
    Image resize to (320,320)
    """
    # parser.add_argument("--size", "--image_size", default=(320,320))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/DCA1/img"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/DCA1/gt"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/DCA1/10_90/S5_best"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/DCA1/DCA1_train_l_10.txt") # DCA1_train_l_5 DCA1_train_l_10
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/DCA1/DCA1_train_u_90.txt") # DCA1_train_u_95 DCA1_train_u_90 
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/DCA1/DCA1_val_34.txt")
    
    parser.add_argument("--lr", "--learning_rate", default=1e-1)
    # parser.add_argument("--l2", "--weight_decay", default=1e-4)
    args = parser.parse_args()
    
    from datetime import datetime
    begin_time = datetime.now()
    main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)