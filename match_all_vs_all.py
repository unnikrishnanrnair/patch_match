import os
from tqdm import tqdm
from natsort import natsorted 

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import optim, nn, utils
import pytorch_lightning as pl
import torchvision as tv
from collections import OrderedDict

from model import SiameseNet, TripletLossModel, Classifier, ClassifierArcface
from dataset_utils.dataset_test import PatchMatchTest
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def main(drone_img_name, map_img_name, model_name):
    print(drone_img_name, 'vs', map_img_name, 'with', model_name)
    image_names_drone = os.listdir('patches_matching_data/test/patches/')
    image_names_map = os.listdir('patches_matching_data/train/patches/')
    
    UAV_images = [x for x in image_names_drone if drone_img_name in x]
    Map_images = [x for x in image_names_map if map_img_name in x]

    UAV_images = natsorted(UAV_images)
    Map_images = natsorted(Map_images)

    drone_dataset = PatchMatchTest(UAV_images, 'patches_matching_data/test/patches/')
    drone_loader = utils.data.DataLoader(
        drone_dataset, batch_size=64,
        shuffle=False, pin_memory=True, num_workers=8
    )
    map_dataset = PatchMatchTest(Map_images, 'patches_matching_data/train/patches/')
    map_loader = utils.data.DataLoader(
        map_dataset, batch_size=64,
        shuffle=False, pin_memory=True, num_workers=8
    )

    UAV_embeddings = np.zeros((len(UAV_images), 1000))
    Map_embeddings = np.zeros((len(Map_images), 1000))

    if model_name=='Classifier':
        model = Classifier()
        batch_size = 64
        state_dict_model = torch.load('model_saves/classifier_version_0/epoch=22-step=13202.ckpt')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.backbone
        model = model.cuda().half()
    elif model_name=='ClassifierArcface':
        model = ClassifierArcface()
        batch_size = 64
        state_dict_model = torch.load('model_saves/classifier_arcface_version_0/epoch=23-step=13776.ckpt')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.backbone
        model = model.cuda().half()
    elif model_name=='TripletLossModel':
        model = TripletLossModel()
        batch_size = 64
        state_dict_model = torch.load('model_saves/triplet_version_0/epoch=24-step=14350.ckpt')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.backbone
        model = model.cuda().half()
    elif model_name=='SiameseNet':
        model = SiameseNet()
        batch_size = 64
        state_dict_model = torch.load('model_saves/siamese_version_0/epoch=7-step=4592.ckpt')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            name = k.replace("model.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.backbone
        model = model.cuda().half()

    with torch.no_grad():
        for i, images in tqdm(enumerate(drone_loader),
                            total=len(drone_loader)):
            images = images.cuda().half()
            outputs = model(images)
            outputs = outputs.data.cpu().numpy()
            UAV_embeddings[
                i*batch_size:(i*batch_size + batch_size), :
            ] = outputs
        for i, images in tqdm(enumerate(map_loader),
                            total=len(map_loader)):
            images = images.cuda().half()
            outputs = model(images)
            outputs = outputs.data.cpu().numpy()
            Map_embeddings[
                i*batch_size:(i*batch_size + batch_size), :
            ] = outputs

    distances = pairwise_distances(UAV_embeddings, Map_embeddings)
    distances = distances.reshape(24480, 120, 204)
    
    if not os.path.isdir('results/'+drone_img_name+'_vs_'+map_img_name+'_with_'+model_name+'/'):
        os.makedirs('results/'+drone_img_name+'_vs_'+map_img_name+'_with_'+model_name+'/')
    for i in range(len(distances)):
        plt.imsave('results/'+drone_img_name+'_vs_'+map_img_name+'_with_'+model_name+'/'+str(i)+'_heatmap.png', 1.-normalize(distances[i]), cmap='hot', format='png')

    pos=0
    for x in range(24480):
        search_index = x
        retrieved_h, retrieved_w = np.unravel_index(distances[search_index].argmin(), distances[search_index].shape)
        search_h, search_w = (search_index)//204, (search_index)%204
        if (abs(search_h-retrieved_h)+abs(search_w-retrieved_w)) < 20:
            pos+=1
    print(pos/24480)

if __name__ == "__main__":
    drone_image_name = ['UAV_am', 'UAV_noon', 'UAV_pm']
    map_image_name = ['Bing', 'ESRI', 'Yandex']
    models = ['SiameseNet', 'TripletLossModel', 'Classifier', 'ClassifierArcface']
    for x in drone_image_name:
        for y in map_image_name:
            for z in models:
                main(x, y, z)
