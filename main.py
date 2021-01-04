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

from model import *
from loader import *



def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(
        0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" %
               batches_done, nrow=n_row, normalize=True)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=500,
                        help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=14,
                        help="number of classes for dataset")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)
    cuda = True if torch.cuda.is_available() else False

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    light_train = CustomDatasetFromCSV("train", img_transform)
    # custom_mnist_from_csv.__getitem__(10)
    # test_g=ModelD()
    # x=torch.zeros((1,29,259))
    # l=torch.zeros((1,14))
    # test_g.forward(x,l)
    train_loader = DataLoader(light_train,
                              batch_size=opt.batch_size, shuffle=True)

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = ModelG(opt.latent_dim)
    discriminator = ModelD()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(
            np.random.randint(0, opt.n_classes, batch_size)))
        print(gen_labels.shape,batch_size,gen_labels)
        # Generate a batch of images

        gen_label_onehot = torch.FloatTensor(opt.batch_size, 14)
        gen_label_onehot = gen_label_onehot.cuda()
        gen_label_onehot.resize_(batch_size, 14).zero_()
        gen_label_onehot.scatter_(1, gen_labels.view(batch_size, 1), 1)
        gen_label_onehot = Variable(gen_label_onehot)
        print("before gen, z:{} gen_label:{}".format(z.shape,gen_label_onehot.shape))
        generator.forward(z,gen_label_onehot)
        gen_imgs = generator(z, gen_label_onehot)
        print("####",gen_imgs.shape)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_label_onehot)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_onehot = torch.FloatTensor(batch_size, 14)
        real_onehot = real_onehot.cuda()
        real_onehot.resize_(batch_size, 14).zero_()
        real_onehot.scatter_(1, labels.view(batch_size, 1), 1)
        real_onehot=Variable(real_onehot)

        validity_real = discriminator(real_imgs, real_onehot)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_label_onehot)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

# parser = argparse.ArgumentParser('Conditional DCGAN')
# parser.add_argument('--batch_size', type=int, default=16,
#                     help='Batch size (default=16)')
# parser.add_argument('--lr', type=float, default=0.0001,
#                     help='Learning rate (default=0.01)')
# parser.add_argument('--epochs', type=int, default=10,
#                     help='Number of training epochs.')
# parser.add_argument('--nz', type=int, default=600,
#                     help='Number of dimensions for input noise.')
# parser.add_argument('--cuda', action='store_true',
#                     help='Enable cuda')
# parser.add_argument('--save_every', type=int, default=1,
#                     help='After how many epochs to save the model.')
# parser.add_argument('--print_every', type=int, default=5,
#                     help='After how many epochs to print loss and save output samples.')
# parser.add_argument('--save_dir', type=str, default='models',
#                     help='Path to save the trained models.')
# parser.add_argument('--samples_dir', type=str, default='samples',
#                     help='Path to save the output samples.')
# args = parser.parse_args()
# if not os.path.exists(args.save_dir):
#     os.mkdir(args.save_dir)

# if not os.path.exists(args.samples_dir):
#     os.mkdir(args.samples_dir)

# INPUT_SIZE = 29*259
# SAMPLE_SIZE = 28
# NUM_LABELS = 14

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
# light_train = CustomDatasetFromCSV("train", img_transform)
# # custom_mnist_from_csv.__getitem__(10)
# # test_g=ModelD()
# # x=torch.zeros((1,29,259))
# # l=torch.zeros((1,14))
# # test_g.forward(x,l)
# train_loader = DataLoader(light_train,
#                           batch_size=args.batch_size, shuffle=True)

# model_d = ModelD()
# model_g = ModelG(args.nz)
# criterion = nn.BCELoss()
# input = torch.FloatTensor(args.batch_size, INPUT_SIZE)
# noise = torch.FloatTensor(args.batch_size, (args.nz))

# fixed_noise = torch.FloatTensor(SAMPLE_SIZE, args.nz).normal_(0, 1)
# fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
# for i in range(NUM_LABELS):
#     for j in range(SAMPLE_SIZE // NUM_LABELS):
#         fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

# label = torch.FloatTensor(args.batch_size)
# one_hot_labels = torch.FloatTensor(args.batch_size, 14)

# if args.cuda:
#     model_d.cuda()
#     model_g.cuda()
#     input, label = input.cuda(), label.cuda()
#     noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
#     one_hot_labels = one_hot_labels.cuda()
#     fixed_labels = fixed_labels.cuda()

# optim_d = optim.SGD(model_d.parameters(), lr=args.lr)
# optim_g = optim.SGD(model_g.parameters(), lr=args.lr)
# fixed_noise = Variable(fixed_noise)
# fixed_labels = Variable(fixed_labels)

# real_label = 1
# fake_label = 0

# for epoch_idx in range(args.epochs):
#     model_d.train()
#     model_g.train()
#     d_loss = 0.0
#     g_loss = 0.0
#     for batch_idx, (train_x, train_y) in enumerate(train_loader):
#         batch_size = (train_x.size(0))
#         train_x = train_x.view(-1, INPUT_SIZE)
#         if args.cuda:
#             train_x = train_x.cuda()
#             train_y = train_y.cuda()
#         #TRAIN DISCRIMINATOR
#         input.resize_as_(train_x).copy_(train_x)
#         label.resize_(batch_size).fill_(real_label)

#         one_hot_labels.resize_(batch_size, 14).zero_()
#         one_hot_labels.scatter_(1, train_y.view(batch_size, 1), 1)
#         inputv = Variable(input)
#         labelv = Variable(label)

#         output = model_d(train_x, Variable(one_hot_labels))
#         optim_d.zero_grad()
#         errD_real = criterion(output, labelv)
#         errD_real.backward()
#         realD_mean = output.data.cpu().mean()

#         one_hot_labels.zero_()
#         rand_y = torch.from_numpy(
#             np.random.randint(0, NUM_LABELS, size=(batch_size,1))).cuda()
#         one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
#         noise.resize_(batch_size, args.nz).normal_(0,1)
#         label.resize_(batch_size).fill_(fake_label)
#         noisev = Variable(noise)
#         labelv = Variable(label)
#         onehotv = Variable(one_hot_labels)
#         g_out = model_g(noisev, onehotv)
#         output = model_d(g_out, onehotv)
#         errD_fake = criterion(output, labelv)
#         fakeD_mean = output.data.cpu().mean()
#         errD = errD_real + errD_fake
#         errD_fake.backward()
#         optim_d.step()

#         #train generator
#         noise.normal_(0, 1)
#         one_hot_labels.zero_()
#         rand_y = torch.from_numpy(
#             np.random.randint(0, NUM_LABELS, size=(batch_size, 1))).cuda()
#         one_hot_labels.scatter_(1, rand_y.view(batch_size, 1), 1)
#         label.resize_(batch_size).fill_(real_label)
#         onehotv = Variable(one_hot_labels)
#         noisev = Variable(noise)
#         labelv = Variable(label)
#         g_out = model_g(noisev, onehotv)
#         output = model_d(g_out, onehotv)
#         errG = criterion(output, labelv)
#         optim_g.zero_grad()
#         errG.backward()
#         optim_g.step()

#         d_loss += errD.data
#         g_loss += errG.data
#         if batch_idx % args.print_every == 0:
#             print(
#                 "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
#                 format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
#                        realD_mean))

#             g_out = model_g(fixed_noise, fixed_labels).data.view(
#                 SAMPLE_SIZE, 1, 29, 259).cpu()
#             save_image(g_out,
#                        '{}/{}_{}.png'.format(
#                            args.samples_dir, epoch_idx, batch_idx))

#     print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
#                                                                d_loss, g_loss))
#     if epoch_idx % args.save_every == 0:
#         torch.save({'state_dict': model_d.state_dict()},
#                    '{}/model_d_epoch_{}.pth'.format(
#             args.save_dir, epoch_idx))
#         torch.save({'state_dict': model_g.state_dict()},
#                    '{}/model_g_epoch_{}.pth'.format(
#             args.save_dir, epoch_idx))
