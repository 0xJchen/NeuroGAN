import numpy as np
import argparse
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import nn, optim
import matplotlib.pyplot as plt
from model import *
from loader import *
import pickle
latent_dim = 500
batch_size = 16
# load data
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
light_train = CustomDatasetFromCSV("train", img_transform)

# scaler = MinMaxScaler(feature_range=(0, 1))

# scaler.fit(light_train.data)

train_loader = DataLoader(light_train,
                          batch_size=batch_size, shuffle=True)


generator = ModelG(latent_dim)
discriminator = ModelD()

generator.cuda()
discriminator.cuda()

# generate z

generator.load_state_dict(torch.load(
'./model/model_g_epoch_45.pth')['state_dict'])

z = Variable(torch.cuda.FloatTensor(
    np.random.normal(0, 1, (1, latent_dim))))
# labels = np.array([num for num in range(14)])
stimulus=[]
for i in range(14):

    labels = (torch.tensor(np.array([i])))

    gen_label_onehot = torch.zeros(1, 14)
    # gen_label_onehot = gen_label_onehot
    gen_label_onehot.scatter_(1, labels.view(1, 1), 1)
    gen_label_onehot=gen_label_onehot.cuda()
    gen_imgs = generator(z, gen_label_onehot)
    # print(gen_imgs.shape)
    gen_imgs = gen_imgs.reshape(1, 29*259)
    dat=scaler.inverse_transform(gen_imgs.cpu().data)
    dat = dat.reshape(29, 259)
    plt.matshow(dat)
    info = "validate/"+"_label_"+str(i)
    plt.savefig(info)
    stimulus.append(dat)

pickle.dump(stimulus, open("stimulus", "wb"))
