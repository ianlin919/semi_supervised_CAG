from .efficientunet import Effi_UNet
from .enet import ENet
from .pnet import PNet2D
from .unet import UNet, UNet_DS, UNet_URPC, UNet_CCT, UNet_UniMatch, UNet_DTC
import argparse
from .vision_transformer import SwinUnet as ViT_seg
from .config import get_config
from .mcnet import MCNet2d_v1, MCNet2d_v2
from .frunet import FR_UNet
from .unext import UNext, UNext_S
from .cmunet import CMUNet
from .proj_unet import Proj_UNet
from .pcps import PCPS, PCPS2
from.ske_net import UnetAttention
from .unetpp import UNet_2Plus
from .unet3p import UNet_3Plus
from .attunet import AttU_Net
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Supervision_CNN_Trans2D', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="./networks/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet2p":
        net = UNet_2Plus(in_channels=in_chns, n_classes=1, is_ds=True).cuda()
    elif net_type == "unet3p":    
        net = UNet_3Plus(in_channels=class_num, n_classes=1,).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_unimatch":
        net = UNet_UniMatch(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_dtc":
        net = UNet_DTC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnetv1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnetv2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', 
                        # encoder_weights='imagenet',
                        encoder_weights=None,
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "ViT_Seg":
        net = ViT_seg(config, img_size=args.patch_size,
                      num_classes=args.num_classes).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "attunet":
        net = AttU_Net(img_ch=in_chns, output_ch=class_num).cuda()
    elif net_type == "fr_unet":
        net = FR_UNet(num_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unext":
        net = UNext(num_classes=class_num, input_channels=in_chns)
    elif net_type == "cmu_net":
        net = CMUNet(img_ch=in_chns, output_ch=class_num)
    elif net_type == "unet_proj":
        net = Proj_UNet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "pcps":
        net = PCPS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "pcps2":
        net = PCPS2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == 'ske':
        net = UnetAttention()
    else:
        net = None
    return net