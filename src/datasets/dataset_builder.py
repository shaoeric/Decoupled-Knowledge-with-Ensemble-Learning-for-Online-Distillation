try:
    from .cifar import *
    from .tinyimagenet import TinyImageNet
except:
    from sys import path
    path.append('../datasets')
    from cifar import *
    from tinyimagenet import TinyImageNet

import os

import yaml
from easydict import EasyDict
import re

__all__ = [
    "DatasetBuilder"
]


DATASET_CONFIG = os.path.join(os.path.dirname(__file__), "dataset_config.yml")


def parse_dataset_config():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(DATASET_CONFIG, 'r') as f:
        data = yaml.load(f, Loader=loader)
    return EasyDict(data)


class DatasetBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(dataset_name: str = 'CIFAR10',
             *args, **kwargs):
        config = parse_dataset_config()[dataset_name]
        config.update(kwargs)
        dataset = globals()[dataset_name](*args, **config)
        return dataset, {dataset_name: config}


if __name__ == '__main__':
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, 4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])

    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]
    
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset, trainset_config = DatasetBuilder.load(dataset_name="TinyImageNet", transform=train_transform, train=True, model_num=3)
    valset, valset_config = DatasetBuilder.load(dataset_name="TinyImageNet", transform=val_transform, train=False, class2label=trainset.class_to_idx, model_num=3)
    print(trainset_config)
    print(valset_config)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)
    for img, label in tqdm(trainloader):
        # print(img.shape, label.shape)  # torch.Size([16, 3, 32, 32]) torch.Size([16])
        img1 = img[0, 0].cpu().numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
        img2 = img[0, 1].cpu().numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
        img3 = img[0, 2].cpu().numpy().transpose(1, 2, 0) * np.array(std) + np.array(mean)
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.subplot(1, 3, 2)
        plt.imshow(img2)
        plt.subplot(1, 3, 3)
        plt.imshow(img3)
        plt.title(trainset.classes[label[0].item()])
        plt.show()
        