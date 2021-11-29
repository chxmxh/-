#注释已经写在代码文档里了
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

path_image = "./inputs/Nikon_006_HR.png"
path_base = "<directory_of_pt>/RealSR_EDSR_X4_base.pt"
path_moa = "<directory_of_pt>/RealSR_EDSR_X4_moa.pt"

state_base = torch.load(path_base, map_location=lambda storage, loc: storage)
state_moa = torch.load(path_moa, map_location=lambda storage, loc: storage)
net_base.load_state_dict(state_base)
net_moa.load_state_dict(state_moa)

LR = io.imread(path_image)
LR_tensor = im2tensor(LR).unsqueeze(0).to(dev)

with torch.no_grad():
    SR_base = tensor2im(net_base(LR_tensor))
    SR_moa = tensor2im(net_moa(LR_tensor))


    LR_plot = LR[100:400, 750:1100] / 255
SR_base_plot = SR_base[100:400, 750:1100] / 255
SR_moa_plot = SR_moa[100:400, 750:1100] / 255

diff_SR_base = (LR_plot-SR_base_plot).mean(2) * 10
diff_SR_moa = (LR_plot-SR_moa_plot).mean(2) * 10

f, axarr = plt.subplots(3, 2, figsize=(10, 12))
axarr[0, 0].imshow(LR_plot)
axarr[0, 0].set_title("Input (HR)", fontsize=18)
axarr[0, 0].axis("off")
 
axarr[0, 1].axis("off")
 
axarr[1, 0].imshow(SR_base_plot)
axarr[1, 0].set_title("EDSR w/o MoA", fontsize=18)
axarr[1, 0].axis("off")
 
axarr[1, 1].imshow(diff_SR_base, vmin=0, vmax=1, cmap="viridis")
axarr[1, 1].set_title("EDSR w/o MoA (Δ)", fontsize=18)
axarr[1, 1].axis("off")

axarr[2, 0].imshow(SR_moa_plot)
axarr[2, 0].set_title("EDSR w/ MoA", fontsize=18)
axarr[2, 0].axis("off")
 
axarr[2, 1].imshow(diff_SR_moa, vmin=0, vmax=1, cmap="viridis")
axarr[2, 1].set_title("EDSR w/ MoA (Δ)", fontsize=18)
axarr[2, 1].axis("off")

plt.show()