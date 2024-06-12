## Basic
import os
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import ssl
from utils import *
## DeepLearning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import  lightning.pytorch as pl
from lightning.fabric import Fabric
## Model
from networks import net_factory
import segmentation_models_pytorch as smp
## Data & Augmentation
from dataset import ImageDataSet_Valid
## Initial
ssl._create_default_https_context = ssl._create_unverified_context

def create_model(norm=True, ema=False):
    # net =  smp.Unet("timm-resnest101e", in_channels=1, classes=1)
    net = net_factory('unet',1,1) # unet_cct unet_urpc unet_unimatch mcnetv1 mcnetv2
    # 'unet','enet','pnet'
    if norm:
        model = kaiming_normal_init_weight(net)
    else:
        model = net
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
    
def test(fabric, model, valid_loader, save_dir, use_tta=False, name = 'best.pt'):
    model_path = os.path.join(save_dir, name)
    model.load_state_dict(torch.load(model_path))
    if use_tta:
        from pseudoaug import PseudoAug
        model = PseudoAug(model=model)
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
    metrics = {
        'f1':[], 
        'dice':[], 
        'precision':[], 
        'recall':[], 
        'specificity':[], 
        'iou':[], 
        'acc':[],
        'auc':[]
    }
    metrics_medpy = {
        'dice':[],
        'hd95':[],
        'asd':[],
    }
    from torchvision.utils import save_image
    with torch.no_grad():
        with tqdm(total=len(valid_loader.dataset), desc="test ", unit="img",
                    bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
            for idx, (imgs, labels) in enumerate(valid_loader):
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
                # test metrics
                _f1, _dice, _precision, _recall, _specificity, _iou, _acc = get_metrics(preds, labels)
                metrics["acc"].append(_acc)
                metrics["dice"].append(_dice)
                metrics["f1"].append(_f1)
                metrics["precision"].append(_precision)
                metrics["recall"].append(_recall)
                metrics["specificity"].append(_specificity)
                metrics["iou"].append(_iou)
                try:
                    _auc = get_metrics_auc(preds, labels)
                    metrics["auc"].append(_auc)
                except:
                    pass
                # medpy metrics
                try:
                    medpy_dice, medpy_hd95, medpy_asd = get_metrics_medpy(preds, labels)
                    metrics_medpy["dice"].append(medpy_dice)
                    metrics_medpy["hd95"].append(medpy_hd95)
                    metrics_medpy["asd"].append(medpy_asd)
                except:
                    pass
                pbar.update(imgs.shape[0]) 
                # save result to file
                preds = torch.sigmoid(preds)
                preds = preds > 0.5
                preds = preds.to(torch.float32)
                labels = labels == torch.max(labels)
                labels = labels.to(torch.float32)
                save_image(torch.cat((imgs, labels, preds)), os.path.join(save_dir, 'result_all_{}.png'.format(idx)))
                save_image(labels, os.path.join(save_dir, 'result_gt_{}.png'.format(idx)))
                save_image(preds, os.path.join(save_dir, 'result_pred_{}.png'.format(idx)))
                save_image(imgs, os.path.join(save_dir, 'result_raw_{}.png'.format(idx)))
    
    logs = {'valid_dice_score': v_F1 / len(valid_loader),
            'valid_precision': v_PR / len(valid_loader),
            'valid_sensitivity': v_SE / len(valid_loader),
            'valid_specificity': v_SP / len(valid_loader),
            }
    print(logs)
    print(
        " Valid Accuracy     : {}±{}\n".format(np.array(metrics['acc']).mean(), np.array(metrics['acc']).std()),
        "Valid Dice Score   : {}±{}\n".format(np.array(metrics['dice']).mean(), np.array(metrics['dice']).std()),
        "Valid F1 Score     : {}±{}\n".format(np.array(metrics['f1']).mean(), np.array(metrics['f1']).std()),
        "Valid Precision    : {}±{}\n".format(np.array(metrics['precision']).mean(), np.array(metrics['precision']).std()),
        "Valid Recall       : {}±{}\n".format(np.array(metrics['recall']).mean(), np.array(metrics['recall']).std()),
        "Valid Specificity  : {}±{}\n".format(np.array(metrics['specificity']).mean(), np.array(metrics['specificity']).std()),
        "Valid IoU          : {}±{}\n".format(np.array(metrics['iou']).mean(), np.array(metrics['iou']).std()),
        "Valid medpy_dice   : {}±{}\n".format(np.array(metrics_medpy['dice']).mean(), np.array(metrics_medpy['dice']).std()),
        "Valid medpy_hd95   : {}±{}\n".format(np.array(metrics_medpy['hd95']).mean(), np.array(metrics_medpy['hd95']).std()),
        "Valid medpy_asd    : {}±{}\n".format(np.array(metrics_medpy['asd']).mean(), np.array(metrics_medpy['asd']).std()),
        "Valid AUC          : {}±{}\n".format(np.array(metrics['auc']).mean(), np.array(metrics['auc']).std()),
    )

def test_cam(fabric, model, valid_loader, save_dir):
    result_cam = []
    from grad_cam import show_cam_on_image, SemanticSegmentationTarget, GradCAM
    model_path = os.path.join(save_dir,'best.pt')
    model.load_state_dict(torch.load(model_path))
    # from lightning.fabric import is_wrapped
    # if not is_wrapped(model):
    #     model = fabric.setup(model)
    # if not is_wrapped(valid_loader):
    #     valid_loader = fabric.setup_dataloaders(valid_loader)
    print(model.encoder.down4.maxpool_conv[-1])
    target_layers = [model.encoder.down4.maxpool_conv[-1]]
    # model.train()
    # with torch.no_grad():
    with tqdm(total=len(valid_loader.dataset), desc="test_cam ", unit="img",
                bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as pbar:
        for imgs, labels in valid_loader:
            imgs, labels = imgs.float(), labels.float()
            model, imgs, labels = model.cpu(), imgs.cpu(), labels.cpu()
            preds = model(imgs)
            targets = [SemanticSegmentationTarget(0, labels)]
            rgb_img = torch.permute(imgs[0], (1,2,0)).detach().cpu().numpy()
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            grayscale_cam = cam(input_tensor=imgs,targets=targets)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            result_cam.append(visualization)
            pbar.update(imgs.shape[0]) 
            break
    result_cam = np.array(result_cam)
    np.save('./test_cam.npy', result_cam)    
    np.save('./test_pred.npy', np.array([imgs.detach().numpy(), labels.detach().numpy(), preds.detach().numpy()]))    
    return 

def main(args):
    moving_dot_product = torch.empty(1)
    limit = 3.0 ** (0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)
    
    train_history = createTrainHistory(["loss", "SE", "SP", "PR", "F1"])
    fabric = Fabric(accelerator="gpu", 
                    devices=[args.g],
                    precision="16-mixed",)
    fabric.launch()
    """
    Setup Model, Optimizer, Scheduler, Loss
    Option: pytorch 2.0 compile model
    """
    model = create_model(norm=True)
    # model = torch.compile(model)
    print("=" * 50)
    print("model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))
    print("=" * 50)
    # model = fabric.setup(model)
    """
    Setup Dataset and Dataloader
    """
    valid_dataset = ImageDataSet_Valid(txt_path=args.v_txt_path,
                                 img_dir=args.i_dir,
                                 label_dir=args.l_dir,
                                 size=args.size,
                                 sort=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              num_workers=args.c,
                              shuffle=False,
                              pin_memory=True)
    # valid_loader = fabric.setup_dataloaders(valid_loader)
    
    test(fabric, model, valid_loader, args.s_dir, args.tta, name="best.pt")
    
    # test_cam(fabric, model, valid_loader, args.s_dir)
    
    return

if __name__ == "__main__":
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser.add_argument("--c", "--cpu_core", default=4, type=int)
    parser.add_argument("--g", "--gpu", default=0, type=int)
    """
    FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation
    https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1
    FIVES Dataset Labeled 800, train/test 600/200(official split)
    Image resize from (2048, 2048) to (512,512)
    """
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/FIVES/600/supervised")) # 30 60 120 180 600
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/FIVES/img"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/FIVES/gt"))
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/FIVES/FIVES_valid_200.txt")
    """
    ISIC2018 Dataset Labeled 2594
    Dataset split to train and valid : 2075/519
    Image resize to (512,512)
    """
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/ISIC2018/46/415_1660/S3_best_simaug")) # 208_1867 415_1660
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/ISIC2018/images"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/ISIC2018/masks/0"))
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/ISIC2018/ISIC2018_val_46_519.txt")
    """
    DCA1 Dataset Labeled 134
    Dataset split to train and valid : 100/34
    Image resize to (320,320)
    """
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/DCA1/20_80/CRAUPP"))
    # parser.add_argument("--size", "--image_size", default=(320,320))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/DCA1/img"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/DCA1/gt"))
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/DCA1/DCA1_val_34.txt")
    """
    BUSI Dataset Labeled 647 (only benign and mailgnant images)
    Dataset split to train and valid : 517/130
    by using sklearn train_test_split seed 46
    Image resize to (256,256)
    """
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/BUSI/517/superdived"))
    # parser.add_argument("--size", "--image_size", default=(256,256))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/busi/images"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/busi/masks/0"))
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/busi/BUSI_val_46_130.txt")
    """
    STARE Dataset Labeled 20 and Unlabeled 377
    Dataset split to train and valid : 18/2
    Image resize to (704,704)
    """
    parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/STARE/F6/18/unet_simaug2"))
    parser.add_argument("--size", "--image_size", default=(704,704))
    parser.add_argument("--i_dir", "--img_dir", default=Path("./data/stare/original"))
    parser.add_argument("--l_dir", "--label_dir", default=Path("./data/stare/label"))
    parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/stare/val6.txt")
    """
    CAG Dataset Labeled 500 and Unlabeled 8952
    Dataset split to train and valid : 400/100
    Image resize to (512,512)
    """
    # parser.add_argument("--s_dir", "--save_dir", default=Path("./logs/CAG/F2/400_8952/CPS"))
    # parser.add_argument("--size", "--image_size", default=(512,512))
    # parser.add_argument("--i_dir", "--img_dir", default=Path("./data/cag/imgs"))
    # parser.add_argument("--l_dir", "--label_dir", default=Path("./data/cag/labels"))
    # parser.add_argument("--v_txt_path", "--valid_txt_path", default="./data/cag/valid_2.txt")
    
    parser.add_argument("--tta", help="test time augmentation",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()
    
    from datetime import datetime
    begin_time = datetime.now()
    main(args)
    finish_time = datetime.now()
    print("*" * 150)
    print(f"training end at {finish_time}")
    print(f"Total Training Time : {finish_time - begin_time}")
    print("*" * 150)