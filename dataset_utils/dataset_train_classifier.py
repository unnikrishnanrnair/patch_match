import os
import random
import PIL
from PIL import Image
import pandas as pd
from natsort import natsorted 

import torch
import torchvision as tv
from torch.utils.data import Dataset

class PatchMatchTrainClassifier(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_names = os.listdir(self.img_dir)
        # self.image_names  = natsorted(self.image_names)
        self.transforms = tv.transforms.Compose([
                          tv.transforms.Resize((224,224)),
                          tv.transforms.RandomVerticalFlip(),
                          tv.transforms.RandomHorizontalFlip(),
                          tv.transforms.ColorJitter(brightness=.5, hue=.5, saturation=.5, contrast=.5),
                        #   tv.transforms.Grayscale(num_output_channels=3),
                          tv.transforms.ToTensor(),
                          tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])
                          ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_x = int(image_name.split('_')[1])
        image_y = int(image_name.split('_')[2].split('.')[0])
        class_of_map = (image_x)*204 + (image_y)
        img = Image.open(self.img_dir+image_name)
        return  self.transforms(img), torch.tensor(class_of_map)

def main():
    dataset = PatchMatchTrainClassifier('patches_matching_data/train/patches/')
    for x in range(1000):
        print(dataset.__getitem__(x)[1])

if __name__ == "__main__":
    main()