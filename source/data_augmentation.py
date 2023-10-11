""" This script will help us to extract the data.zip and realize
    data augmentation actions to the database
"""
import os
import yaml
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from typing import Text
import imgaug.augmenters as iaa

 ##------- 2.Define the data_augmentation function ----------##
def data_aug(image_name,n_aug,indx,seq): 
        '''His funtion will help us to read each image  and apply a number of certain augmentation techniques

            1. Brigthness
            2. Color Temperature
            3. Saturation
            4. Blur

            Will save all the augmentations in the selected folder
        '''
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

def data_augmentation(config_path: Text) -> None:
    # read the parameters in the params.yaml file
    params = yaml.safe_load(open(config_path))["data_augmentation"]

    ##---- 1. Define the data transformations into a list -------#
    seq = [
        iaa.WithBrightnessChannels(iaa.Add((params['WithBrightnessChannels'][0], params['WithBrightnessChannels'][1]))),
        iaa.ChangeColorTemperature((params['ChangeColorTemperature'][0], params['ChangeColorTemperature'][1])),
        iaa.AddToSaturation((params['AddToSaturation'][0], params['AddToSaturation'][1])),
        iaa.RemoveSaturation(mul=params['RemoveSaturation_mul']),
        iaa.MotionBlur(k=params['MotionBlur_k'])
        ] 

    ##-------------- 3. Data Augmentatation process ------------------##
    images_path = params['images_path']
    images_name = []
    n_aug = 2
    print('generando nuevas imágenes....')
    with os.scandir(images_path) as ficheros:
        for fichero in ficheros:
            names = os.listdir(fichero)
            lastname = names[len(names)-1]
            indx = int(lastname.split(".")[1])+1

            for name in names:
                image_name = os.path.join(images_path,fichero.name,name)
                data_aug(image_name,n_aug,indx+1,seq=seq)
                indx+=n_aug
                break
    print('hecho!')

    shutil.make_archive('data_aug','zip','./data/')
    print('zip generado')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    data_augmentation(config_path=args.params)