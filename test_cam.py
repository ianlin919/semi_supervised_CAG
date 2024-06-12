import cv2
import warnings
from grad_cam import ClassifierOutputTarget, show_cam_on_image, GradCAM
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import numpy as np
label1 = cv2.imread('/home/jannawu/Desktop/S3/1229/train/imgs/CVAI-0541_LAD_RAO25_CRA27_25.png', cv2.IMREAD_UNCHANGED)
label1 = cv2.cvtColor(label1, cv2.COLOR_GRAY2RGB)
from copy import deepcopy
rgb_img = deepcopy(label1)/255
label1 = np.expand_dims(label1, 0).astype(np.float32)
label1 = torch.from_numpy(label1)
label1 = torch.permute(label1,(0,-1,1,2))
input_tensor = label1
from torchvision.models import resnet50
model = resnet50(pretrained=True)
# model = torch.compile(model)
output = model(input_tensor)
target_layers = [model.layer4[-1]]
from copy import deepcopy
rgb_img = deepcopy(label1)/255
rgb_img = torch.permute(rgb_img[0], (1,2,0)).detach().cpu().numpy()
targets = [ClassifierOutputTarget(1)]
input_tensor= input_tensor/255
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
grayscale_cam = cam(input_tensor=input_tensor,targets=targets)[0]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
np.save('./cam.npy',cam_image)