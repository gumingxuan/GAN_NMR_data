import argparse
import math
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

print(torch.__version__)
print(torch.version.cuda)
# # 初始化参数
for i in range(0, 1):
    print('Initialization parameters')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()