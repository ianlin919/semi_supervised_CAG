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
from dataset import ImageDataSet0, ImageDataSet, ImageDataSet1, ImageDataSet4
from transforms import get_train_augmentation
from transforms.RandAugment import RandAugment_best_2aug
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=True, ema=False):
    # net = smp.Unet("efficientnet-b6", in_channels=1, classes=1)
    # timm-resnest101e
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
        model_path = os.path.join(save_dir, 'best.pt')
        torch.save(model.state_dict(), model_path)
        best_target = current_target
        best_epoch = epoch
        return best_target, best_epoch
    else:
        return best_target, best_epoch

def train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch):
    model.train()
    su_loss = 0
    with tqdm(total=len(train_loader.dataset), desc="train", unit="img",
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
        with tqdm(total=len(valid_loader.dataset), desc="test ", unit="img",
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
    return logs
def select_pesudo_label(fabric, model, valid_loader, save_dir, step_model_name):
    model_all = []
    from lightning.fabric import is_wrapped
    from copy import deepcopy
    if not is_wrapped(valid_loader):
            valid_loader = fabric.setup_dataloaders(valid_loader)
    for name in step_model_name:
        model_path = os.path.join(save_dir, name)
        model.load_state_dict(torch.load(model_path))
        if not is_wrapped(model):
            model_step = fabric.setup(model)
        model.eval()
        model_step = deepcopy(model)
        model_all.append(model_step)
    threshold = 0.5
    file_index = 0
    all_result_list = []
    with torch.no_grad():
        with tqdm(total=len(valid_loader.dataset), desc="pseudo ", unit="img",
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            for imgs in valid_loader:
                imgs = imgs.float()
                preds1 = model_all[0](imgs)
                preds2 = model_all[1](imgs)
                preds3 = model_all[2](imgs)
                mask = (torch.sigmoid(preds3).ge(threshold)).to(dtype=torch.float32)
                F1 = (get_F1(preds1, mask) + get_F1(preds2, mask))/2
                file_name = valid_loader.dataset.fileNames[file_index]
                all_result_list.append([file_index, file_name, F1])
                file_index = file_index + 1
                pbar.update(imgs.shape[0])
    del model_all
    all_result_list = sorted(all_result_list,key=lambda x: x[2],reverse=True)
    all_result_list = np.array(all_result_list, dtype=object)
    
    reliable_list = all_result_list[len(all_result_list)//2:]
    unreliable_list = all_result_list[:len(all_result_list)//2]
    np.savetxt(os.path.join(save_dir, "reliable_list.csv"), reliable_list, delimiter=",", fmt='%s')
    np.savetxt(os.path.join(save_dir, "unreliable_list.csv"), unreliable_list, delimiter=",", fmt='%s')
    np.savetxt(os.path.join(save_dir, "name_reliable.txt"), reliable_list.T[1], fmt='%s')
    np.savetxt(os.path.join(save_dir, "name_unreliable.txt"), unreliable_list.T[1], fmt='%s')
    return all_result_list, reliable_list, unreliable_list, "name_reliable.txt", "name_unreliable.txt"

def pseudo_labeling(fabric, model, valid_loader, save_dir, pesudo_list):
    threshold = 0.5
    file_index = 0
    pseudo_index = pesudo_list.T[0]
    model_path = os.path.join(save_dir,'best.pt')
    model.load_state_dict(torch.load(model_path))
    from lightning.fabric import is_wrapped
    if not is_wrapped(model):
        model = fabric.setup(model)
    if not is_wrapped(valid_loader):
        valid_loader = fabric.setup_dataloaders(valid_loader)
    model.eval()
    from torchvision.utils import save_image
    with torch.no_grad():
        with tqdm(total=len(valid_loader.dataset), desc="pseudo ", unit="img",
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            for imgs in valid_loader:
                imgs = imgs.float()
                preds = model(imgs)
                mask = (torch.sigmoid(preds).ge(threshold)).to(dtype=torch.float32)
                result = mask[0]
                if file_index in pseudo_index:
                    file_name = valid_loader.dataset.fileNames[file_index]
                    save_image(result, save_dir/file_name)
                file_index = file_index + 1
                pbar.update(imgs.shape[0])
    return

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
    # model = torch.compile(model)
    print("=" * 50)
    print("model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    print("=" * 50)
    # loss = GeneralizedDiceLoss()
    # loss = nn.BCEWithLogitsLoss()
    loss = DiceBCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=args.lr, 
                                #  weight_decay=args.l2
                                 )
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=600)
    model, optimizer = fabric.setup(model, optimizer)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet0(txt_path=args.t_txt_path,
                                  img_dir=args.i_dir,
                                  label_dir=args.l_dir,
                                  size=args.size,
                                  transform=get_train_augmentation())
    valid_dataset = ImageDataSet(txt_path=args.v_txt_path,
                                 img_dir=args.i_dir,
                                 label_dir=args.l_dir,
                                 size=args.size,
                                 sort=True)
    valid_dataset_u = ImageDataSet1(txt_path=args.t_un_txt_path,
                                    img_dir=args.i_dir,
                                    size=args.size,
                                    sort=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.bl,
                              num_workers=args.c,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=args.c,
                              shuffle=False,
                              pin_memory=True)
    valid_loader_u = DataLoader(dataset=valid_dataset_u,
                                batch_size=1,
                                num_workers=args.c,
                                shuffle=False,
                                pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    valid_loader_u = fabric.setup_dataloaders(valid_loader_u)
    
    best_target = 0
    best_epoch = 0
    step_model_name=[]
    for epoch in range(args.e):
        train_history = train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch)
        train_history, current_target = valid(fabric, model, valid_loader, epoch, train_history)
        best_target, best_epoch = save_model(fabric, model, best_target, current_target, args.s_dir, epoch, best_epoch)
        print(f"Epoch {epoch + 1} "
              f"loss : {train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history['valid']['F1'][epoch]:2.5f}")
        print("Best F1/DSC {} on Epoch {}".format(str(best_target),str(best_epoch)))
        if epoch == args.e//3 or epoch == args.e*2//3 or epoch == args.e*3//3-1:
            step_name = 'step_e_{}.pt'.format(str(epoch))
            model_path = os.path.join(args.s_dir, step_name)
            step_model_name.append(step_name)
            torch.save(model.state_dict(), model_path)
    fabric.logger.save()
    print(step_model_name)
    saveDict(args.s_dir/'train_history.pickle', train_history)
    train_sup_log = test(fabric, model, valid_loader, args.s_dir)
    all_result_list, reliable_list, unreliable_list, name_reliable, name_unreliable = select_pesudo_label(fabric, model, valid_loader_u, args.s_dir, step_model_name)
    pseudo_labeling(fabric, model, valid_loader_u, args.s_dir, reliable_list)
    return train_sup_log, all_result_list, reliable_list, unreliable_list, name_reliable, name_unreliable

def main_st(args, name_reliable, name_unreliable, unreliable_list):
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
    # model = torch.compile(model)
    print("=" * 50)
    print("model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    print("=" * 50)
    # loss = GeneralizedDiceLoss()
    # loss = nn.BCEWithLogitsLoss()
    loss = DiceBCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=args.lr, 
                                #  weight_decay=args.l2
                                 )
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=600)
    model, optimizer = fabric.setup(model, optimizer)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet4(txt_path=args.t_txt_path,
                                  un_txt_path=os.path.join(args.s_dir, name_reliable),
                                  img_dir=args.i_dir,
                                  label_dir=args.l_dir,
                                  pseudo_dir=args.s_dir,
                                  size=args.size,
                                  transform=get_train_augmentation())
    valid_dataset = ImageDataSet(txt_path=args.v_txt_path,
                                 img_dir=args.i_dir,
                                 label_dir=args.l_dir,
                                 size=args.size,
                                 sort=True)
    valid_dataset_u = ImageDataSet1(txt_path=os.path.join(args.s_dir, name_unreliable),
                                    img_dir=args.i_dir,
                                    size=args.size,
                                    sort=True)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.bu,
                            #   num_workers=args.c,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=args.c,
                              shuffle=False,
                              pin_memory=True)
    valid_loader_u = DataLoader(dataset=valid_dataset_u,
                                batch_size=1,
                                num_workers=args.c,
                                shuffle=False,
                                pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    valid_loader_u = fabric.setup_dataloaders(valid_loader_u)
    best_target = 0
    best_epoch = 0
    
    for epoch in range(args.e):
        train_history = train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch)
        train_history, current_target = valid(fabric, model, valid_loader, epoch, train_history)
        best_target, best_epoch = save_model(fabric, model, best_target, current_target, args.s_dir, epoch, best_epoch)
        print(f"Epoch {epoch + 1} "
              f"loss : {train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history['valid']['F1'][epoch]:2.5f}")
        print("Best F1/DSC {} on Epoch {}".format(str(best_target),str(best_epoch)))
    fabric.logger.save()
    saveDict(args.s_dir/'train_history_st.pickle', train_history)
    train_st_log = test(fabric, model, valid_loader, args.s_dir)
    pseudo_labeling(fabric, model, valid_loader_u, args.s_dir, unreliable_list)
    return train_st_log

def main_stp(args):
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
    # model = torch.compile(model)
    print("=" * 50)
    print("model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    print("=" * 50)
    # loss = GeneralizedDiceLoss()
    # loss = nn.BCEWithLogitsLoss()
    loss = DiceBCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=args.lr, 
                                #  weight_decay=args.l2
                                 )
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=600)
    model, optimizer = fabric.setup(model, optimizer)
    """
    Setup Dataset and Dataloader
    """
    train_dataset = ImageDataSet4(txt_path=args.t_txt_path,
                                  un_txt_path=args.t_un_txt_path,
                                  img_dir=args.i_dir,
                                  label_dir=args.l_dir,
                                  pseudo_dir=args.s_dir,
                                  size=args.size,
                                  transform=get_train_augmentation())
    valid_dataset = ImageDataSet(txt_path=args.v_txt_path,
                                 img_dir=args.i_dir,
                                 label_dir=args.l_dir,
                                 size=args.size,
                                 sort=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.bu,
                            #   num_workers=args.c,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=args.c,
                              shuffle=False,
                              pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)
    valid_loader = fabric.setup_dataloaders(valid_loader)
    
    best_target = 0
    best_epoch = 0
    
    for epoch in range(args.e):
        train_history = train(fabric, model, train_loader, loss, optimizer, scheduler, train_history, epoch)
        train_history, current_target = valid(fabric, model, valid_loader, epoch, train_history)
        best_target, best_epoch = save_model(fabric, model, best_target, current_target, args.s_dir, epoch, best_epoch)
        print(f"Epoch {epoch + 1} "
              f"loss : {train_history['train']['loss'][epoch]:2.5f}, "
              f"valid_PR : {train_history['valid']['PR'][epoch]:2.5f}, "
              f"valid_SE : {train_history['valid']['SE'][epoch]: 2.5f}, "
              f"valid_SP : {train_history['valid']['SP'][epoch]:2.5f}, "
              f"valid_F1 : {train_history['valid']['F1'][epoch]:2.5f}")
        print("Best F1/DSC {} on Epoch {}".format(str(best_target),str(best_epoch)))
    fabric.logger.save()
    saveDict(args.s_dir/'train_history_stp.pickle', train_history)
    train_stp_log = test(fabric, model, valid_loader, args.s_dir)
    return train_stp_log

if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument("--bl", "--batch_size_labeled", default=4, type=int)
    parser.add_argument("--bu", "--batch_size_un", default=10, type=int)
    parser.add_argument("--e", "--Epoch", default=10, type=int)
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=3, type=int)
    """
    CAG Dataset Labeled 500 and Unlabeled 8952
    Dataset split to train and valid : 400/100
    Image resize to (512,512)
    """
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/cag/imgs"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/cag/labels"))
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/STPP/F2/20_500/best2")) #s_add_augment pseudoaug_ada
    # # # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/cag/labeled_400_2.txt")
    # # # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/cag/unlabeled_all.txt")
    # # # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
    
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
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/STARE/STPP/18_377"))
    # parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/stare/train3.txt")
    # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled.txt")
    # # parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/stare/unlabeled_clean.txt")
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/stare/val3.txt")
    """
    DCA1 Dataset Labeled 134
    Dataset split to train and valid : 100/34
    Image resize to (320,320)
    """
    parser.add_argument("--size", "--image_size", default=(320,320))
    parser.add_argument("--i_dir", "--img_dir", default=Path("./data/DCA1/img"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("./data/DCA1/gt"))
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/DCA1/5_95/STPP"))
    parser.add_argument("--t_txt_path", "--train_txt_path", default="./data/DCA1/DCA1_train_l_5.txt") # DCA1_train_l_5 DCA1_train_l_10
    parser.add_argument("--t_un_txt_path", "--train_un_txt_path", default="./data/DCA1/DCA1_train_u_95.txt") # DCA1_train_u_95 DCA1_train_u_90 
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/DCA1/DCA1_val_34.txt")
    
    parser.add_argument("--lr", "--learning_rate", default=1e-3)
    # parser.add_argument("--l2", "--weight_decay", default=1e-4)
    args = parser.parse_args()
    
    from datetime import datetime
    begin_time = datetime.now()
    train_sup_log, all_result_list, reliable_list, unreliable_list, name_reliable, name_unreliable = main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)
    
    begin_time = datetime.now()
    train_st_log = main_st(args, name_reliable, name_unreliable, unreliable_list)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)
    
    begin_time = datetime.now()
    train_stp_log = main_stp(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)
    
    print("Train Supervised Metrics", train_sup_log)
    print("Train Self-Training Metrics", train_st_log)
    print("Train Self-Training ++ Metrics", train_stp_log)