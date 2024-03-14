import argparse
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.__version__)
print(torch.version.cuda)

#
data_csv = pd.read_csv('Data_for_presentation.csv')
data = np.array(data_csv.values.tolist())
core_num = len(data)
T2s, T2c = data[:, 2:130], data[:, 130:]
zeros_matrix = np.zeros((core_num, 128))
true_data = np.concatenate((zeros_matrix, T2s, zeros_matrix, T2c, zeros_matrix), axis=1)
true_data1 = true_data.reshape(core_num, 1, 5, 128)
true_data = np.concatenate((T2s, T2c), axis=1)
true_data2 = true_data.reshape(core_num, 256)
true_data3 = []
for i in range(1, core_num + 1):  # Assuming there are a total of core_num images
    image_path = os.path.join(os.getcwd() + '\\true_data3', f'Fig_type_3.jpg')
    img = Image.open(image_path)
    img_array = np.array(img)
    true_data3.append(img_array)
    img.close()
true_data3 = np.array(true_data3)
#
true_data3 = true_data3.transpose(0, 3, 1, 2)
#
true_data3 = true_data3[:, 0, :, :]
true_data3 = true_data3.reshape(true_data3.shape[0], 1, true_data3.shape[1], true_data3.shape[2])
image_shape1 = true_data1[0].shape
image_shape2 = true_data2[0].shape
image_shape3 = true_data3[0].shape

#
for i in range(0, 1):
    print('plot data')
    plt.imshow(np.squeeze(true_data1[0], axis=0), cmap='viridis', aspect='auto')
    plt.axis('off')
    plt.show()
    #
    plt.plot(np.arange(1, 129), true_data2[0, :128], marker='o', linestyle='-', color='red')
    plt.plot(np.arange(129, 257), true_data2[0, 128:], marker='o', linestyle='-', color='green')
    plt.axis('off')
    plt.show()
    #
    plt.imshow(np.squeeze(true_data3[0], axis=0), cmap='gray')
    plt.axis('off')
    plt.show()

class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(256, 5 * 128),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img1 = self.model(z)
        img1 = img1.view(img1.size(0), *image_shape1)
        return img1

class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img2 = self.model(z)
        img2 = img2.view(img2.size(0), *image_shape2)
        return img2

class Generator3(nn.Module):
    def __init__(self):
        super(Generator3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(512, 5120),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(5120, 131*175),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img3 = self.model(z)
        img3 = img3.view(img3.size(0), *image_shape3)
        return img3

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 1 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img1):
        validity = self.model(img1)
        return validity

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img2):
        validity = self.model(img2)
        return validity

class Discriminator3(nn.Module):
    def __init__(self):
        super(Discriminator3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 17 * 22, 1),
            nn.Sigmoid()
        )

    def forward(self, img3):
        validity = self.model(img3)
        return validity

