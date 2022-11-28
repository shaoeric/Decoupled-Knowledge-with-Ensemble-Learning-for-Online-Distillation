import os
import torch
from torchvision.datasets import DatasetFolder
from PIL import Image


class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, classes2label=None, *args, **kwargs):
        super(TestImageDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.model_num = kwargs['model_num']
        self.classes2label = classes2label
        self.image_file_list, self.label_list = self.parse_txt()

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        file = os.path.join(self.root,'images', self.image_file_list[idx])
        img = Image.open(file).convert('RGB')
        label = torch.tensor(int(self.label_list[idx])).long()

        if self.transform is not None:
            img_list = []
            for _ in range(self.model_num):
                img_transform = self.transform(img.copy())
                img_list.append(img_transform)
            img = torch.stack(img_list)

        return img, label

    def parse_txt(self):
        annotation_path = os.path.join(self.root, 'val_annotations.txt')
        image_file_list = []
        label_list = []

        with open(annotation_path, 'r') as f:
            contents = f.readlines()
        for content in contents:
            image_file, classes_name = content.split('\t')[:2]
            image_file_list.append(image_file)
            label = self.classes2label[classes_name]
            label_list.append(label)
        return image_file_list, label_list

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path: str):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            model_num=None,
            loader=default_loader,
            is_valid_file=None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.model_num = model_num

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img_list = []
            for _ in range(self.model_num):
                img_transform = self.transform(img.copy())
                img_list.append(img_transform)
            img = torch.stack(img_list)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)


def TinyImageNet(root, transform, train=True, class2label=None, model_num=None):
    assert model_num is not None
    if train:
        train_folder = os.path.join(root, 'train')
        dataset = ImageFolder(train_folder, transform=transform, model_num=model_num)
    elif class2label is not None:
            val_folder = os.path.join(root, 'val')
            dataset = TestImageDataset(root=val_folder, transform=transform, classes2label=class2label, model_num=model_num)
    else:
        raise NotImplementedError
    return dataset