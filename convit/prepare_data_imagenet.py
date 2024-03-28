import numpy as np
import torch
from PIL import Image
import os
from tqdm import tqdm

# Path to original dataset
original_path = "/home/haneol.kijm/Works/data/imagenet-mini/train/"

# Path to processed dataset
data_path = "/home/haneol.kijm/Works/data/imagenet-mini_processed/"

# Path to store trained models
models_path = "/home/haneol.kijm/Works/git/imaging_MLPs/trained_networks/"

# Path to store tensor board data
logs_path = "./logs/"


def crop_center(img):
    new_width = 256
    new_height = 256

    width, height = img.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))

    return np.array(img)


def resize_small(img):
    new_width = 224
    new_height = 224

    width, height = img.size  # Get dimensions

    if width < 224 or height < 224:
        img = img.resize((new_width, new_height))

    return np.array(img)


def add_noise(img):
    _, imgh, imgw = img.shape
    noise = np.float32(np.random.normal(0, 30 / 255, (3, imgh, imgw)))
    noisy_img = np.clip(img + noise, 0, 1)

    return noisy_img


# In[9]:


k = 0
l = 0
m = 0
for i in tqdm(os.listdir(original_path)):
    for j in range(11):

        if j < len(os.listdir(original_path + i)):

            img_path = os.listdir(original_path + i)[j]
            img = Image.open(original_path + i + "/" + img_path)
            # img=crop_center(img)   ##crop center to 256x256
            img = resize_small(img)
            if len(img.shape) == 2:  ##check there aren't imgs with 2 channels only
                img = img[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            img = np.transpose(img, (2, 0, 1))  ##reshape to have number of channels first
            img = np.float32(img / np.max(img))  ##convert to float
            noisy_img = add_noise(img)

            if m < 970:
                torch.save(torch.FloatTensor(img), data_path + "clean_train/" + "{0:05}".format(k))
                torch.save(
                    torch.FloatTensor(noisy_img), data_path + "noisy_train/" + "{0:05}".format(k)
                )
                k += 1
            else:
                torch.save(torch.FloatTensor(img), data_path + "clean_val/" + "{0:05}".format(l))
                torch.save(
                    torch.FloatTensor(noisy_img), data_path + "noisy_val/" + "{0:05}".format(l)
                )
                l += 1

    m += 1


print(len(os.listdir(data_path + "clean_train/")))
