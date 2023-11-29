import os
import shutil 
import glob
import random
import argparse
from zipfile import ZipFile
from typing import Text 
import yaml

##----------1.  we will extract all files in data.zip--------------------##
def data_preprocess(config_path: Text) -> None:

    params = yaml.safe_load(open(config_path))["data_preprocess"]
    # specifying the name of the zip file
    filepath = os.path.abspath(params['abspath'])
  
    # open the zip file in read mode
    with ZipFile(filepath, 'r') as zip: 
        # list all the contents of the zip file
        zip.printdir() 
  
        # extract all files
        print('extraction...') 
        zip.extractall() 
        print('Done!')

    ##--2. We will extract images from train folder to create the test set---##

    # define the data folder were we will extract images
    images_path = params['images_path']

    # read the content
    images_name = glob.glob(images_path, 
                   recursive = True)

    # Choice a random number of images inside the data folder
    validation_set = random.choices(images_name,k=30)

    # create the test folder into data dir 
    if not os.path.isdir("data/test"):
        os.mkdir("data/test")
        os.mkdir("data/test/cats")
        os.mkdir("data/test/dogs")

    # Move form data/train to data/test
    for image in validation_set:
        try:
            new_name = image.replace("train","test")
            print(image)
            print(new_name)
            shutil.move(image,new_name)
        except Exception:
            print(f'Image {image} not founded!')
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    data_preprocess(config_path=args.params)