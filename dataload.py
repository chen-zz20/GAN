import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os


transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

class MNIST_Dataset(Dataset):
    def __init__(self, batch_size, path) -> None:
        super().__init__()
        self._training_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transform
        )
    
        self._validation_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transform
        )

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=False,
            pin_memory=True
        )
    
    @property
    def training_data(self) -> Dataset:
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader


class CIFAR10_Dataset(Dataset):
    def __init__(self, batch_size, path) -> None:
        super().__init__()
        self._training_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transform
        )
    
        self._validation_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transform
        )

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=False,
            pin_memory=True
        )
    
    @property
    def training_data(self) -> Dataset:
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader


class CIFAR100_Dataset(Dataset):
    def __init__(self, batch_size, path) -> None:
        super().__init__()
        self._training_data = datasets.CIFAR100(
            root=path,
            train=True,
            download=True,
            transform=transform
        )
    
        self._validation_data = datasets.CIFAR100(
            root=path,
            train=False,
            download=True,
            transform=transform
        )

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=10,
            shuffle=False,
            pin_memory=True
        )
    
    @property
    def training_data(self) -> Dataset:
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader


def Choose_Dataset(batch_size:int, path:str, mode:str="MNIST") -> Dataset:
    if not os.path.isdir(path):
        os.mkdir(path)
    
    if mode == "MNIST":
        return MNIST_Dataset(batch_size, path)
    elif mode == "CIFAR10":
        return CIFAR10_Dataset(batch_size, path)
    elif mode == "CIFAR100":
        return CIFAR100_Dataset(batch_size, path)
    else:
        exit("mode should be in ['MNIST, 'CIFAR10', 'CIFAR100']")