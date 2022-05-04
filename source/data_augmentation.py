""" This script will help us to extract the data.zip and realize
    data augmentation actions to the database
"""
import os
import shutil
from PIL import Image
from pathlib2 import Path
import imgaug.augmenters as iaa
import numpy as np
import random

##---- 1. Define the data transformations into a list -------#
seq = [
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.ChangeColorTemperature((1100, 10000)),
    iaa.AddToSaturation((-40, 40)),
    iaa.RemoveSaturation(mul=0.25),
    iaa.MotionBlur(k=3)
    ]

##------- 2.Define the data_augmentation function ----------##
def data_aug(image_name,n_aug,indx): 
    img = Image.open(image_name)
    aug_c = random.sample(seq,n_aug)
    middle_name = image_name.split(".")
   
    for aug in aug_c:
        new_image = aug(image = np.array(img))
        final_name = f'{middle_name[0]}.{str(indx).zfill(4)}.{middle_name[-1]}'
        indx +=1
        new_image = Image.fromarray(new_image)
        new_image.save(final_name)
    return True   

##-------------- 3. Data Augmentatation process ------------------##
images_path = "data/train/"
images_name = []
n_aug = 2
with os.scandir(images_path) as ficheros:
    for fichero in ficheros:
        names = os.listdir(fichero)
        lastname = names[len(names)-1]
        indx = int(lastname.split(".")[1])+1

        for name in names:
            image_name = os.path.join(images_path,fichero.name,name)
            data_aug(image_name,n_aug,indx+1)
            indx+=n_aug
            break
           