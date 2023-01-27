import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Module
from torchvision.utils import make_grid, save_image
from torch.optim import Optimizer
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from model import Generator, Discriminator

import os
from tqdm import tqdm


class Trainer(object):
    def __init__(self, netG:Generator, netD:Discriminator, optimG:Optimizer, optimD:Optimizer, dataset:Dataset, train_dir:str, tb_writer:SummaryWriter, notes:str="test", device:torch.device=None) -> None:
        self.netG = netG
        self.netD = netD
        self.optimG = optimG
        self.optimD = optimD
        self.dataset = dataset
        self.train_dir = train_dir
        self.tb_writer = tb_writer
        self.notes = notes

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(train_dir, exist_ok=True)
        self.netG.load(train_dir, "")
        self.netD.load(train_dir, "")
    
    def train_step(self, real_imgs:Tensor, fake_imgs:Tensor, BCE_criterion:Module) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """DO NOT FORGET TO ZERO_GRAD netD and netG
        *   Returns:
            *   loss of netD (scalar)
            *   loss of netG (scalar)
            *   average D(real_imgs) before updating netD
            *   average D(fake_imgs) before updating netD
            *   average D(fake_imgs) after updating netD
        """
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # clear gradients
        self.netD.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(real_imgs), 1) w.r.t. netD
        # record average D(real_imgs)
        x_D = self.netD(real_imgs)
        loss_D_real = BCE_criterion(x_D, torch.ones_like(x_D, device=self.device))
        D_real = x_D.mean()
        loss_D_real.backward()

        # ** accumulate ** the gradients of binary_cross_entropy(netD(fake_imgs), 0) w.r.t. netD
        # record average D(fake_imgs)
        z_D = self.netD(fake_imgs)
        loss_D_fake = BCE_criterion(z_D, torch.zeros_like(z_D, device=self.device))
        D_fake_1 = z_D.mean()
        loss_D_fake.backward(retain_graph=True)

        # update netD
        self.optimD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # clear gradients
        self.netG.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(fake_imgs), 1) w.r.t. netG
        # record average D(fake_imgs)
        z_G = self.netD(fake_imgs)
        loss_G = BCE_criterion(z_G, torch.ones_like(z_G, device=self.device))
        D_fake_2 = z_G.mean()
        loss_G.backward()

        # update netG
        self.optimG.step()

        # return what are specified in the docstring
        return loss_D_real + loss_D_fake, loss_G, D_real, D_fake_1, D_fake_2
    
    def train(self, num_epochs:int, saving_epochs:int) -> None:
        fixed_noise = torch.randn(50, self.netG.latent_dim, 1, 1, device=self.device)
        cirterion = nn.BCELoss()
        training_loader = self.dataset.training_loader
        for epoch in tqdm(range(1, 1 + num_epochs), desc="Training"):
            self.netD.train()
            self.netG.train()
            loss_D, loss_G, D_real, D_fake_1, D_fake_2 = 0.0, 0.0, 0.0, 0.0, 0.0
            times = 0
            for step, (real_imgs, tragets) in enumerate(training_loader):
                real_imgs = real_imgs.to(self.device)
                fake_noise = torch.randn(real_imgs.size(0), self.netG.latent_dim, 1, 1, device=self.device)
                fake_imgs = self.netG(fake_noise)

                loss_D_, loss_G_, D_real_, D_fake_1_, D_fake_2_ = self.train_step(real_imgs, fake_imgs, cirterion)

                loss_D += loss_D_.data.cpu().numpy()
                loss_G += loss_G_.data.cpu().numpy()
                D_real += D_real_.data.cpu().numpy()
                D_fake_1 += D_fake_1_.data.cpu().numpy()
                D_fake_2 += D_fake_2_.data.cpu().numpy()
                times += 1
            
            loss_D /= times
            loss_G /= times
            D_real /= times
            D_fake_1 /= times
            D_fake_2 /= times
            
            self.tb_writer.add_scalar("discriminator_loss", loss_D, global_step=epoch)
            self.tb_writer.add_scalar("generator_loss", loss_G, global_step=epoch)
            self.tb_writer.add_scalar("D_real", D_real, global_step=epoch)
            self.tb_writer.add_scalar("D_fake_1", D_fake_1, global_step=epoch)
            self.tb_writer.add_scalar("D_fake_2", D_fake_2, global_step=epoch)

            if epoch % saving_epochs == 0:
                dirname = self.netD.save(self.train_dir, notes=os.path.join(self.notes, str(epoch)))
                dirname = self.netG.save(self.train_dir, notes=os.path.join(self.notes, str(epoch)))
                self.netG.eval()
                imgs = make_grid(self.netG(fixed_noise)) * 0.5 + 0.5
                self.tb_writer.add_image('samplis', imgs, global_step=epoch)
                save_image(imgs, os.path.join(dirname, "samples.png"))