import numpy as np
import os, torch, random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# fastmri
import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data



def pair_random_crop(inp, tar, image_size):
    # h, w = tar.shape[1], tar.shape[2]
    # padw = image_size - w if w < image_size else 0
    # padh = image_size - h if h < image_size else 0
    # # Reflect Pad in case image is smaller than image_size
    # if padw != 0 or padh != 0:
    #     print(inp.shape, padh, padw)
    #     inp = F.pad(inp, (0, 0, padw, padh), padding_mode="reflect")
    #     print(inp.size())
    #     tar = F.pad(tar, (0, 0, padw, padh), padding_mode="reflect")

    h2, w2 = tar.shape[1], tar.shape[2]
    # print("after: ", tar.shape)
    r = random.randint(0, h2 - image_size)
    c = random.randint(0, w2 - image_size)

    inp = inp[:, r : r + image_size, c : c + image_size]
    tar = tar[:, r : r + image_size, c : c + image_size]
    return inp, tar


def pair_random_flip(inp, tar):
    axis = random.randint(1, 2)
    inp = inp.flip(axis)
    tar = tar.flip(axis)
    return inp, tar


def pair_random_rot(inp, tar):
    angle = random.randint(0, 3)
    inp = torch.rot90(inp, dims=(1, 2), k=angle)
    tar = torch.rot90(tar, dims=(1, 2), k=angle)
    return inp, tar


class RandomTransform(object):
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def __call__(self, inp, tar):
        inp, tar = pair_random_crop(inp, tar, self.image_size)
        if random.random() > 0.5:
            inp, tar = pair_random_flip(inp, tar)
        if random.random() > 0.5:
            inp, tar = pair_random_rot(inp, tar)
        # h, w = inp.shape
        # if h != self.output_size[0] or w != self.output_size[1]:
        return inp, tar


class PairCenterCrop(object):
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def __call__(self, inp, tar):
        inp = F.center_crop(inp, self.image_size)
        tar = F.center_crop(inp, self.image_size)
        return inp, tar


class PairDataset(Dataset):
    def __init__(self, img_dir, noisy_dir, split, transform=None):
        self.split = split
        self.img_dir = img_dir
        self.noisy_dir = noisy_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.split == "train":
            img_path = os.path.join(self.img_dir, "{0:05}".format(idx))
            noisy_path = os.path.join(self.noisy_dir, "{0:05}".format(idx))
            image = torch.load(img_path)
            noisy = torch.load(noisy_path)
        else:
            img_path = os.path.join(self.img_dir, "{0:05}".format(idx))
            noisy_path = os.path.join(self.noisy_dir, "{0:05}".format(idx))
            image = torch.load(img_path)
            noisy = torch.load(noisy_path)
        if self.transform:
            image, noisy = self.transform(image, noisy)
        return image, noisy


