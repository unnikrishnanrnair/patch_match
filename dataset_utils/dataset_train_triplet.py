import os
import random
import PIL
from PIL import Image
import pandas as pd

import torch
import torchvision as tv
from torch.utils.data import Dataset

class PatchMatchTrainTriplet(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_names = os.listdir(self.img_dir)
        self.transforms = tv.transforms.Compose([
                          tv.transforms.Resize((224,224)),
                          tv.transforms.RandomVerticalFlip(),
                          tv.transforms.RandomHorizontalFlip(),
                          tv.transforms.ColorJitter(brightness=.5, hue=.5, saturation=.5, contrast=.5),
                          tv.transforms.ToTensor(),
                          tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])
                          ])
        self.dataset_names = ['Bing', 'ESRI', 'Yandex']

    def __len__(self):
        return len(self.image_names)
    
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
        
        img_anchor = image_name
        img_pos_easy = image_name
        img_pos_hard = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(image_x) + '_' + str(image_y)+'.jpeg'
        new_image_x, new_image_y = self.get_hard_neighbour(image_x, image_y)
        img_neg_hard = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(new_image_x) + '_' + str(new_image_y)+'.jpeg'
        new_image_x, new_image_y = self.get_random_patch(image_x, image_y)
        image_neg_easy = random.choice([x for x in self.dataset_names if x!=image_name.split('_')[0]]) + '_' + str(new_image_x) + '_' + str(new_image_y)+'.jpeg'
        # print(img_anchor, img_pos_easy, img_pos_hard, image_neg_easy, img_neg_hard)

        img_anchor = Image.open(self.img_dir+img_anchor)
        img_pos_easy = Image.open(self.img_dir+img_pos_easy)
        img_pos_hard = Image.open(self.img_dir+img_pos_hard)
        img_neg_hard = Image.open(self.img_dir+img_neg_hard)
        image_neg_easy = Image.open(self.img_dir+image_neg_easy)

        return self.transforms(img_anchor),random.choice([self.transforms(img_pos_easy), self.transforms(img_pos_hard)]),random.choice([self.transforms(img_neg_hard), self.transforms(image_neg_easy)])

def main():
    dataset = PatchMatchTrainTriplet('patches_matching_data/train/patches/')
    for x in range(100):
        print(len(dataset.__getitem__(x)))

if __name__ == "__main__":
    main()