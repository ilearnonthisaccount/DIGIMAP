import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import copy
from util import *
from PIL import Image

from model import *
import moviepy.video.io.ImageSequenceClip
import scipy
import kornia.augmentation as K

from base64 import b64encode
import gradio as gr
from torchvision import transforms

# torch.hub.download_url_to_file('https://i.imgur.com/HiOTPNg.png', 'mona.png')
# torch.hub.download_url_to_file('https://i.imgur.com/Cw8HcTN.png', 'painting.png')

device = 'cpu'
latent_dim = 8
n_mlp = 5
num_down = 3

G_A2B = Generator(256, 4, latent_dim, n_mlp, channel_multiplier=1, lr_mlp=.01,n_res=1).to(device).eval()

ensure_checkpoint_exists('GNR_checkpoint_full.pt')
ckpt = torch.load('GNR_checkpoint_full.pt', map_location=device)

G_A2B.load_state_dict(ckpt['G_A2B_ema'])

# mean latent
truncation = 1
with torch.no_grad():
    mean_style = G_A2B.mapping(torch.randn([1000, latent_dim]).to(device)).mean(0, keepdim=True)


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
])
plt.rcParams['figure.dpi'] = 200

# torch.manual_seed(84986)

num_styles = 1
style = torch.randn([num_styles, latent_dim]).to(device)


def inference(input_im):
    real_A = test_transform(input_im).unsqueeze(0).to(device)

    with torch.no_grad():
        A2B_content, _ = G_A2B.encode(real_A)
        #fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles,1,1,1), style)
        fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles,1,1,1), torch.randn([num_styles, latent_dim]).to(device))
        std=(0.5, 0.5, 0.5)
        mean=(0.5, 0.5, 0.5)
        z = fake_A2B * torch.tensor(std).view(3, 1, 1)
        z = z + torch.tensor(mean).view(3, 1, 1)
        tensor_to_pil = transforms.ToPILImage(mode='RGB')(z.squeeze())
    return tensor_to_pil

title = "GANsNRoses"
article = "<p style='text-align: center'>GANs N' Roses: Image to Iamge Translation | Obtained from :<a href='https://github.com/mchong6/GANsNRoses'>Github Repo</a></p>"
demo = gr.Interface(
    inference, 
    [gr.inputs.Image(type="pil", label="Input")], 
    gr.outputs.Image(type="pil"),
    title=title,
    # description=description,
    article=article,
    allow_flagging = "never",
    )
    
demo.launch(share = True)
