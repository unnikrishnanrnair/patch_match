import os 

import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_and_save_image(root_folder, image_name, img, X,Y,W,H, stride):
    cropped_image = img[X:X+H, Y:Y+W]
    assert (cropped_image.shape[0] == W) and (cropped_image.shape[1] == H)
    assert (X%stride==0) and (Y%stride==0)
    cv2.imwrite(root_folder+'patches/'+image_name.split('.')[0]+'_'+str(X//stride)+'_'+str(Y//stride)+'.jpeg', cropped_image)

def main():
    root_folder = 'patches_matching_data/train/'
    images = os.listdir(root_folder)

    patch_size = 512
    stride = 25

    if not os.path.isdir(root_folder+'patches'):
        os.makedirs(root_folder+'patches')

    for image_name in images:
        if 'tif' in image_name:
            original_image = cv2.imread(root_folder+image_name)
            total_w, total_h = original_image.shape[0], original_image.shape[1]
            startx,starty = 0,0
            count = 0
            while((starty+patch_size)<total_h):
                while((startx+patch_size)<total_w):
                    crop_and_save_image(root_folder, image_name, original_image, startx, starty, patch_size, patch_size, stride)
                    startx+=stride
                    count+=1
                startx=0
                starty+=stride
            print('generated', count, 'number of patches for', image_name, 'of patch_size,', patch_size, 'stride,', stride)

if __name__ == "__main__":
    main()
