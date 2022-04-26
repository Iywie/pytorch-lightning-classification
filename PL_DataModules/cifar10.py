import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.data_dir = cfgs['dir']
        self.batch_size_train = cfgs['batch_size_train']
        self.batch_size_val = cfgs['batch_size_val']
        self.num_workers = cfgs['num_workers']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.244, 0.262]),
        ])

        self.cifar10_train = None
        self.cifar10_val = None
        self.cifar10_test = None
        self.cifar10_predict = None

    # def prepare_data(self):
    #     # download
    #     CIFAR10(self.data_dir, train=True, download=True)
    #     CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_val = CIFAR10(self.data_dir, train=False, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size_train, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size_val, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size_val, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size_val, num_workers=self.num_workers)
