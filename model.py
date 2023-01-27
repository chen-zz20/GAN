import torch
import torch.nn as nn
from torch import Tensor
import os

def weights_init(m:nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def get_generator(latent_dim:int, hidden_dim:int, mode:str="MNIST", device:torch.device=None) -> nn.Module:
    model = Generator(latent_dim, hidden_dim, mode, device).to(device)
    model.apply(weights_init)
    return model

def get_discriminator(hidden_dim:int, mode:str="MNIST", device:torch.device=None) -> nn.Module:
    model = Discriminator(hidden_dim, mode, device).to(device)
    model.apply(weights_init)
    return model

class Generator(nn.Module):
    def __init__(self, latent_dim:int, hidden_dim:int, mode:str="MNIST", device:torch.device=None) -> None:
        super().__init__()
        self.mode = mode
        if mode == "MNIST":
            self.num_channels = 1
        elif mode == "CIFAR10" or mode == "CIFAR100":
            self.num_channels = 3
        else:
            exit("mode should be in ['MNIST, 'CIFAR10', 'CIFAR100']")

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            # [latent_dim, 1, 1]
            # layer 1
            nn.ConvTranspose2d(latent_dim, 4 * hidden_dim, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            # [4 * hidden_dim, 4, 4]
            # layer 2
            nn.ConvTranspose2d(4 * hidden_dim, 2 * hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * hidden_dim),
            nn.ReLU(),
            # [2 * hidden_dim, 8, 8]
            # layer 3
            nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # [hidden_dim, 16, 16]
            # layer 4
            nn.ConvTranspose2d(hidden_dim, self.num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z:Tensor) -> Tensor:
        '''
        *   Arguments:
            *   z (torch.FloatTensor): [batch_size, latent_dim, 1, 1]
        '''
        z = z.to(self.device)
        return self.net(z)
    
    def load(self, train_dir:str, notes:str="test"):
        try:
            if os.path.exists(os.path.join(train_dir, 'generator.pth')):
                path = os.path.join(train_dir, 'generator.pth')
            else:
                path = os.path.join(train_dir, notes, 'generator.pth')
        except:
            print("model load error!")
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, train_dir:str, notes:str="test"):
        path = os.path.join(train_dir, notes)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'generator.pth')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]

        
class Discriminator(nn.Module):
    def __init__(self, hidden_dim:int, mode:str="MNIST", device:torch.device=None) -> None:
        super().__init__()
        self.mode = mode
        if mode == "MNIST":
            self.num_channels = 1
        elif mode == "CIFAR10" or mode == "CIFAR100":
            self.num_channels = 3
        else:
            exit("mode should be in ['MNIST, 'CIFAR10', 'CIFAR100']")

        self.hidden_dim = hidden_dim

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            # [num_channels, 32, 32]
            nn.Conv2d(self.num_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [hidden_dim, 16, 16]
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [hidden_dim * 2, 8, 8]
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # [hidden_dim * 4, 4, 4]
            nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x).view(-1, 1).squeeze(1)
    
    def load(self, train_dir:str, notes:str="test"):
        try:
            if os.path.exists(os.path.join(train_dir, 'discriminator.pth')):
                path = os.path.join(train_dir, 'discriminator.pth')
            else:
                path = os.path.join(train_dir, notes, 'discriminator.pth')
        except:
            print("model load error!")
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, train_dir:str, notes:str="test"):
        path = os.path.join(train_dir, notes)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'discriminator.pth')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]