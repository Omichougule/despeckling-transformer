# Code for testing on real SAR images 
# Author: Malsha Perera
import argparse
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as F
import os
import numpy as np
import torch
from transform_main import TransSAR, TransSARV2, TransSARV3
import cv2



parser = argparse.ArgumentParser(description='TransSAR')

parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save_path', required=True , type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', type=str,
                    help='model name')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loadmodel', default='load', type=str)



args = parser.parse_args()


modelname = args.model
loaddirec = args.loadmodel
save_path = args.save_path




device = torch.device("cuda")



model = TransSARV2()


# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model,device_ids=[0,1]).cuda()
# model.to(device)


# model.load_state_dict(torch.load(loaddirec))

# Load the state dictionary
state_dict = torch.load("pretrained_models/model.pth")

# Create a new state dictionary without the "module." prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v  # Remove "module." prefix
    else:
        new_state_dict[k] = v

# Load the new state dictionary into the model
model.load_state_dict(new_state_dict)

model.eval()



if not os.path.isdir(save_path):
                
    os.makedirs(save_path)




im_file = 'test_images/SI_2014-01-04_TSX_7_2_125.png'

img = cv2.imread(im_file,0) 
noisy_im = (np.float32(img)+1.0)/256.0



x = np.float32(noisy_im)
x = F.to_tensor(x)
print(x.shape)

x = x.unsqueeze(0)
from torchvision.transforms import v2
from torchvision.transforms import CenterCrop, ToPILImage
crop_size = 512
pil_image = ToPILImage()(x.squeeze(0))


center_cropped_image = CenterCrop(crop_size)(pil_image)
center_cropped_image_np = np.array(CenterCrop(crop_size)(pil_image))

cv2.imwrite("test_images/SI_2014-01-04_TSX_7_2_125_center.png", center_cropped_image_np)

x = F.to_tensor(center_cropped_image)
print(x.shape)
x = x.unsqueeze(0)
print(x.shape)
pred_im = model(x)
tmp = pred_im.detach().cpu().numpy()

tmp = tmp.squeeze()
tmp = tmp*256 - 1 

filename_out = 'SI_2014-01-04_TSX_7_2_125_results.png'
filepath_out = save_path + filename_out

cv2.imwrite(filepath_out,tmp)


print('done')