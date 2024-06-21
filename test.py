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
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--loadmodel', default='load', type=str)



args = parser.parse_args()


modelname = args.model
loaddirec = args.loadmodel
save_path = args.save_path




device = torch.device("cpu")
map_location=torch.device('cpu')


model = TransSARV2()


# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model,device_ids=[0,1]).cuda()
# model.to(device)


# model.load_state_dict(torch.load(loaddirec))
# model.load_state_dict(torch.load(loaddirec, map_location=torch.device('cpu')))
# model.eval()

state_dict = torch.load(loaddirec, map_location=torch.device('cpu'))

# Create a new state dict with the keys fixed
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')  # remove `module.` prefix
    new_state_dict[new_key] = value

# Load the new state dict into the model
model.load_state_dict(new_state_dict)

# Ensure the model is in evaluation mode
model.eval()



if not os.path.isdir(save_path):
                
    os.makedirs(save_path)




im_file = 'test_images/Pittsburg_CA_New_York_Point.png'

img = cv2.imread(im_file,0) 
noisy_im = (np.float32(img)+1.0)/256.0



x = np.float32(noisy_im)
x = F.to_tensor(x)
x = x.unsqueeze(0)


pred_im = model(x)
tmp = pred_im.detach().cpu().numpy()

tmp = tmp.squeeze()
tmp = tmp*256 -1

filename_out = 'Pittsburg_CA_New_York_Point_results.png'
filepath_out = save_path + filename_out

cv2.imwrite(filepath_out,tmp)


print('done')

