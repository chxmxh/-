path_image_HR = "./inputs/Canon_003_HR.png"
path_image_LR = "./inputs/Canon_003_LR4.png"
path_base = "<directory_of_pt>/RealSR_EDSR_X4_base.pt"
path_moa = "<directory_of_pt>/RealSR_EDSR_X4_moa.pt"

state_base = torch.load(path_base, map_location=lambda storage, loc: storage)
state_moa = torch.load(path_moa, map_location=lambda storage, loc: storage)
net_base.load_state_dict(state_base)
net_moa.load_state_dict(state_moa)

HR = io.imread(path_image_HR)
HR_tensor = im2tensor(HR).unsqueeze(0).to(dev)

LR = io.imread(path_image_LR)
LR_tensor = im2tensor(LR).unsqueeze(0).to(dev)

# apply CutBlur
LR_tensor[..., 900:1250, :600] = HR_tensor[..., 900:1250, :600]

with torch.no_grad():
    SR_base = tensor2im(net_base(LR_tensor))
    SR_moa = tensor2im(net_moa(LR_tensor))
    
LR_plot = tensor2im(LR_tensor)[900:1500, 100:650] / 255
HR_plot = HR[900:1500, 100:650] / 255
SR_base_plot = SR_base[900:1500, 100:650] / 255
SR_moa_plot = SR_moa[900:1500, 100:650] / 255

diff_SR_base = (HR_plot-SR_base_plot).mean(2) * 10
diff_SR_moa = (HR_plot-SR_moa_plot).mean(2) * 10

f, axarr = plt.subplots(3, 2, figsize=(10, 14))
axarr[0, 0].imshow(LR_plot)
axarr[0, 0].set_title("Input (Cutblurred LR)", fontsize=18)
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
