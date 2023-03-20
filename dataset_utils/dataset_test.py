import os
import random
import PIL
from PIL import Image
import pandas as pd

import torch
import torchvision as tv
from torch.utils.data import Dataset

class PatchMatchTest(Dataset):
    def __init__(self, img_list, root_path):
        self.img_list = img_list
        self.root_dir = root_path
        self.transforms = tv.transforms.Compose([
                          tv.transforms.Resize((224,224)),
                        #   tv.transforms.Grayscale(num_output_channels=3),
                          tv.transforms.ToTensor(),
                          tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])
                          ])

    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        img = Image.open(self.root_dir+image_name)
        return self.transforms(img)

    def __len__(self):
        return len(self.img_list)

def main():
    image_names_drone = os.listdir('patches_matching_data/test/patches/')
    image_names_map = os.listdir('patches_matching_data/train/patches/')
    dataset = PatchMatchTest(image_names_drone, 'patches_matching_data/test/patches/')
    for x in range(1000):
        dataset.__getitem__(x)
        print(x)
    dataset = PatchMatchTest(image_names_map, 'patches_matching_data/train/patches/')
    for x in range(1000):
        dataset.__getitem__(x)
        print(x)

if __name__ == "__main__":
    main()