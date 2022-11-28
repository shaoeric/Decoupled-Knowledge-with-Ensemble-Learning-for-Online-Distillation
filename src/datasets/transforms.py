from torchvision.transforms import transforms

__all__ = ['cifar100_transform', 'cifar10_transform', 'tinyimagenet_transform']


def cifar100_transform():
    mean = [0.5071, 0.4866, 0.4409]
    std = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, val_transform


def cifar10_transform():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, val_transform


def tinyimagenet_transform():
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
    return train_transform, val_transform