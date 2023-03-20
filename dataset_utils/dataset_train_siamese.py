import os
import random
import PIL
from PIL import Image
import pandas as pd

import torch
import torchvision as tv
from torch.utils.data import Dataset

class PatchMatchTrain(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_names = os.listdir(self.img_dir)
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
        self.prob_hard_neg = 0.5
        self.prob_hard_pos = 0.99999
        self.prob_positive = 0.5
        self.dataset_names = ['Bing', 'ESRI', 'Yandex']

    def __len__(self):
        return len(self.image_names)
    
    def decision(self, probability):
        return random.random() < probability

    def get_hard_neighbour(self, x, y):
        scaler = random.choice([1,2,3,4,5,6,7,8,9,10])
        scaler_x = random.choice([1, -1])*scaler
        scaler = random.choice([1,2,3,4,5,6,7,8,9,10])
        scaler_y = random.choice([1, -1])*scaler
        newx = x+scaler_x
        newy = y+scaler_y
        if (newx<0):
            newx=x
        if (newx>119):
            newx=x
        if (newy<0):
            newy=y
        if (newy>203):
            newy=y
        if (newx==x) and (newy==y):
            return self.get_hard_neighbour(x, y)
        else:
            return newx, newy
    
    def get_random_patch(self, x, y):
        newx = random.randint(0, 119)
        newy = random.randint(0, 203)
        if (newx==x) and (newy==y):
            return self.get_random_patch(x,y)
        return newx, newy

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_x = int(image_name.split('_')[1])
        image_y = int(image_name.split('_')[2].split('.')[0])
        if self.decision(self.prob_positive):
            if self.decision(self.prob_hard_pos):
                # get a hard positive pair hard here refers to different map
                image_name1 = image_name
                image_name2 = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(image_x) + '_' + str(image_y)+'.jpeg'
                img1 = Image.open(self.img_dir+image_name1)
                img2 = Image.open(self.img_dir+image_name2)
                return self.transforms(img1), self.transforms(img2), torch.tensor(1)
            else:
                # get an easy positive pair easy here refers to same patch from same map
                image_name1 = image_name
                image_name2 = image_name
                img1 = Image.open(self.img_dir+image_name1)
                img2 = Image.open(self.img_dir+image_name2)
                return self.transforms(img1), self.transforms(img2), torch.tensor(1) 
        else:
            if self.decision(self.prob_hard_neg):
                # get a hard negative pair hard here refers to neighbouring patch from random other map (close by patches are harder to distinguish)
                image_name1 = image_name
                new_image_x, new_image_y = self.get_hard_neighbour(image_x, image_y)
                image_name2 = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(new_image_x) + '_' + str(new_image_y)+'.jpeg'
                img1 = Image.open(self.img_dir+image_name1)
                img2 = Image.open(self.img_dir+image_name2)
                return self.transforms(img1), self.transforms(img2), torch.tensor(0)
            else:
                # get an easy negative pair easy here refers to random patch from random other map (very likely to be a very different image)
                image_name1 = image_name
                new_image_x, new_image_y = self.get_random_patch(image_x, image_y)
                image_name2 = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(new_image_x) + '_' + str(new_image_y)+'.jpeg'
                img1 = Image.open(self.img_dir+image_name1)
                img2 = Image.open(self.img_dir+image_name2)
                return self.transforms(img1), self.transforms(img2), torch.tensor(0)

def main():
    dataset = PatchMatchTrain('patches_matching_data/train/patches/')
    for x in range(100):
        print(x)
        dataset.__getitem__(x)

if __name__ == "__main__":
    main()