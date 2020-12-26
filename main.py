import numpy as np
import functools
from PIL import Image
import IPython.display
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import utils


parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
config = vars(parser.parse_args(''))  # use default arguments
config["seed"] = 0
config["resolution"] = utils.imsize_dict["I128_hdf5"]
config["n_classes"] = utils.nclass_dict["I128_hdf5"]
config["G_activation"] = utils.activation_dict["inplace_relu"]
config["D_activation"] = utils.activation_dict["inplace_relu"]
config["G_attn"] = "64"
config["D_attn"] = "64"
config["G_ch"] = 96
config["D_ch"] = 96
config["hier"] = True
config["dim_z"] = 120
config["shared_dim"] = 128
config["G_shared"] = True
config = utils.update_config_roots(config)
config["skip_init"] = True
config["no_optim"] = True
config["device"] = "cuda"
config['batch_size'] = 1

utils.seed_rng(config["seed"])
torch.backends.cudnn.benchmark = True
model = __import__(config["model"])
experiment_name = utils.name_from_config(config)
G = model.Generator(**config).to(config["device"])
utils.count_parameters(G)
import copy

weights_path = "./models_pretrained/G_ema.pth"
state_dict = torch.load(weights_path)
G.load_state_dict(state_dict)
G.eval()
G_batch_size = max(config["G_batch_size"], config["batch_size"])
(z_, y_) = utils.prepare_z_y(
    G_batch_size,
    G.dim_z,
    config["n_classes"],
    device=config["device"],
    fp16=config["G_fp16"],
    z_var=config["z_var"],
)




alphas = torch.linspace(-3, 3, 7)
ims = []
with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 989

for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_)
    ims.append(G(z_save, G.shared(y_),method = 'l_shifty', alpha = alphas[i]).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)




alphas = torch.linspace(3, -3, 7)
ims = []
with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 989

for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_)
    ims.append(G(z_save, G.shared(y_),method = 'l_shiftx', alpha = alphas[i]).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)



alphas = torch.linspace(-3, 3, 7)
ims = []
with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 989

for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_)
    ims.append(G(z_save, G.shared(y_),method = 'l_zoom', alpha = alphas[i]).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)



# inx control the step size of the nl walk.
with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 989

ims = []
z_save = copy.deepcopy(z_)
ims.append(G(z_save, G.shared(y_), method='nl_zoom', alpha=-3, inx=0.1).detach())
ims.append(G(z_save, G.shared(y_), method='nl_zoom', alpha=-2, inx=0.1).detach())
ims.append(G(z_save, G.shared(y_), method='nl_zoom', alpha=-1, inx=0.1).detach())
ims.append(G(z_save, G.shared(y_)).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_zoom', alpha = 1, inx = 0.05).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_zoom', alpha = 2, inx = 0.05).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_zoom', alpha = 3, inx = 0.05).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)



with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 158

ims = []
z_save = copy.deepcopy(z_)
ims.append(G(z_save, G.shared(y_), method='nl_shiftx', alpha=3, inx=0.05).detach())
ims.append(G(z_save, G.shared(y_), method='nl_shiftx', alpha=2, inx=0.05).detach())
ims.append(G(z_save, G.shared(y_), method='nl_shiftx', alpha=1, inx=0.05).detach())
ims.append(G(z_save, G.shared(y_)).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_shiftx', alpha = -1, inx = 0.05).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_shiftx', alpha = -2, inx = 0.05).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_shiftx', alpha = -3, inx = 0.05).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)






with torch.no_grad():
        z_.sample_()
        y_.sample_()
        y_[0] = 158

ims = []
z_save = copy.deepcopy(z_)
ims.append(G(z_save, G.shared(y_), method='nl_shifty', alpha=-2, inx=0.1).detach())
ims.append(G(z_save, G.shared(y_), method='nl_shifty', alpha=-1, inx=0.1).detach())
ims.append(G(z_save, G.shared(y_)).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_shifty', alpha = 2, inx = 0.2).detach())
ims.append(G(z_save, G.shared(y_),method = 'nl_shifty', alpha = 1, inx = 0.2).detach())

image_grid = torchvision.utils.make_grid(
        torch.cat(ims),
        nrow=7
        ,
        normalize=True,
    )
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)




with torch.no_grad():
    z_.sample_()
    y_.sample_()
    y_[0] = 155

ims = []
z_save = copy.deepcopy(z_).clamp_(-1,1)
(G(z_save, G.shared(y_), method='svd_gcircle', alpha=torch.tensor(0.0) , inx=0, inx2=19).detach())

ims = []
z_save = copy.deepcopy(z_).clamp_(-1,1)
ims.append(G(z_save, G.shared(y_)).detach())
a = np.pi # Explore the whole circle not ordered
alphas = torch.arange(0,a,0.6)
for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_).clamp_(-1, 1)
    ims.append(G(z_save, G.shared(y_), method='svd_gcircle', alpha=alphas[i] , inx=0, inx2=19).detach())

image_grid = torchvision.utils.make_grid(
    torch.cat(ims),
    nrow=alphas.shape[0]+1
    ,
    normalize=True,
)
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)




with torch.no_grad():
    z_.sample_()
    y_.sample_()
    y_[0] = 155

ims = []
z_save = copy.deepcopy(z_).clamp_(-1,1)
alphas = torch.arange(-3,3,1)
for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_).clamp_(-1, 1)
    ims.append(G(z_save, G.shared(y_), method='linearin', alpha=alphas[i] , inx=0, inx2=32).detach()) # inx2 here comes for the scale i.e., 8 for the second, 16 for the third and 64 for the upper one.

image_grid = torchvision.utils.make_grid(
    torch.cat(ims),
    nrow=alphas.shape[0]+1
    ,
    normalize=True,
)
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)



with torch.no_grad():
    z_.sample_()
    y_.sample_()
    y_[0] = 155

ims = []
z_save = copy.deepcopy(z_).clamp_(-1,1)
a = 2*np.pi # Explore the whole circle not ordered
alphas = torch.arange(0,a,0.6)
for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_).clamp_(-1, 1)
    ims.append(G(z_save, G.shared(y_), method='greatin', alpha=alphas[i] , inx=0, inx2=4).detach()) # inx2 here comes for the scale i.e., 8 for the second, 16 for the third and 64 for the upper one.

image_grid = torchvision.utils.make_grid(
    torch.cat(ims),
    nrow=alphas.shape[0]+1
    ,
    normalize=True,
)
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)



with torch.no_grad():
    z_.sample_()
    y_.sample_()
    y_[0] = 155

ims = []
z_save = copy.deepcopy(z_).clamp_(-1,1)
a = 2*np.pi # Explore the whole circle not ordered
alphas = torch.arange(0,a,0.2)
for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_).clamp_(-1, 1)
    ims.append(G(z_save, G.shared(y_), method='smallin', alpha=alphas[i] , inx=0, inx2=4).detach()) # inx2 here comes for the scale the options are [4,8,16,32,64]

image_grid = torchvision.utils.make_grid(
    torch.cat(ims),
    nrow=alphas.shape[0]+1
    ,
    normalize=True,
)
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)






with torch.no_grad():
    z_.sample_()
    y_.sample_()
    y_[0] = 155

ims = []
z_save = copy.deepcopy(z_)
ims.append(G(z_save, G.shared(y_)).detach())
ims.append(G(z_save, G.shared(y_) ,method='svd_scircle', alpha=torch.tensor(0.0) , inx=0, inx2=19).detach())

ims = []
ims.append(G(z_save, G.shared(y_)).detach())
a = 0.7148 # read the results and set
alphas = torch.arange(0,a,0.1)
for i in range(alphas.shape[0]):
    z_save = copy.deepcopy(z_).clamp_(-1, 1)
    ims.append(G(z_save, G.shared(y_), method='svd_scircle', alpha=alphas[i] , inx=0, inx2=19).detach())

image_grid = torchvision.utils.make_grid(
    torch.cat(ims),
    nrow=alphas.shape[0]+1
    ,
    normalize=True,
)
image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0) * 255
image_grid_np = np.uint8(image_grid_np)
print("Image Grid Shape: {}".format(np.shape(image_grid_np)))
print("Max pixel value: {}".format(np.max(image_grid_np)))
print("Min pixel value: {}".format(np.min(image_grid_np)))
fi = plt.imshow(image_grid_np)
fi.axes.get_yaxis().set_visible(False)
fi.axes.get_xaxis().set_visible(False)
















