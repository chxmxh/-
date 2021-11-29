#注释已经写进代码文档里了

"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import sys
sys.path.append("../")
import importlib
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import model.edsr as edsr

# options for EDSR
class Opt:
    scale = 4
    num_blocks = 32
    num_channels = 256
    res_scale = 0.1

def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor

def tensor2im(tensor):
    tensor = tensor.detach().squeeze(0)
    im = tensor.clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
    return im

opt = Opt()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_base = edsr.Net(opt).to(dev)
net_moa = edsr.Net(opt).to(dev)

path_image = "./inputs/0869.png"
path_base = "<directory_of_pt>/DIV2K_EDSR_X4_base.pt"
path_moa = "<directory_of_pt>/DIV2K_EDSR_X4_moa.pt"

state_base = torch.load(path_base, map_location=lambda storage, loc: storage)
state_moa = torch.load(path_moa, map_location=lambda storage, loc: storage)
net_base.load_state_dict(state_base)
net_moa.load_state_dict(state_moa)

LR = io.imread(path_image)
LR_tensor = im2tensor(LR).unsqueeze(0).to(dev)

with torch.no_grad():
    SR_base = tensor2im(net_base(LR_tensor))
    SR_moa = tensor2im(net_moa(LR_tensor))