'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import sys
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import Text
from dvclive import Live
from dvclive.keras import DVCLiveCallback

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping



def train_model(config_path: Text) -> None:
    """ This function to establish the network arquitecture
        Using the keras vgg16 pretrained layers and creating
        the classification final block.
    """
    params = yaml.safe_load(open(config_path))["train"]

    path = os.path.abspath(params['abspath_data'])

    # dimensions of our images.
    best_model_path = params['best_model_path']
    train_data_dir = os.path.join(path, 'train')
    validation_data_dir = os.path.join(path,'validation')


    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    train_data_generator = ImageDataGenerator(
        **data_gen_args,
        preprocessing_function=preprocess_input,
    )
    valid_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_generator = train_data_generator.flow_from_directory(
        train_data_dir,
        target_size=(params["image_size"], params["image_size"]),
        batch_size=params["batch_size_training"],
        class_mode=params["class_mode"],
        # categorical genuino [1,0] spoofing [0,1]
        shuffle=True,
    )

    validation_generator = valid_data_generator.flow_from_directory(
        validation_data_dir,
        target_size=(params["image_size"], params["image_size"]),
        batch_size=params["batch_size_validation"],
        class_mode=params["class_mode"],
        shuffle=True,
    )
    #          Creating the full model to the classification task            #

    #   importing the actual keras vgg16 architecture for convolutional layers
    #   pretrained by imagenet dataset
    vgg_conv = VGG16(
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )

    # just for console visualization
    print(vgg_conv.summary())

    # Freezing convolutional layers
    for layer in vgg_conv.layers[:]:
        layer.trainable = False

    # this will be the whole model architecture
    model = Sequential()
    # adding convolutional vgg16 layers
    model.add(vgg_conv)  # [24,24]
    # Flatten convolutional features
    model.add(Flatten())  # [576,] [1,576]
    # Adding the clasification dense layers
    model.add(Dense(
                    params["mid_neurons"],
                    activation=params["mid_activation"]))
    model.add(Dropout(rate=params["drop_rate"]))

    model.add(Dense(
                    params["mid_neurons"],
                    activation=params["mid_activation"]))
    model.add(Dropout(rate=params["drop_rate"]))
    model.add(
        Dense(params["end_neurons"], activation=params["end_activation"])
    )  # output layer [0.7,0.3] [0,0.99]

    print(model.summary())

    model.compile(
        optimizer=params["optimizer"], loss=params["loss"],
        metrics=params["metric"]
    )

    steps_per_epoch_train = len(train_generator)
    steps_per_epoch_valid = len(validation_generator)
    number_epoch = params["number_epoch"]

    #my_callbacks = [EarlyStopping(patience=2), CSVLogger("history.csv")]

    with Live('metrics/') as live:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch_train,
            epochs=number_epoch,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch_valid,
            callbacks=[DVCLiveCallback(live=live)]
        )

    # saving the history to metrics step
    # convert the history.history dict to a pandas DataFrame:     
    metrics_raw = history.history
    metric = dict()

    metric['accuracy'] = metrics_raw['accuracy'][-1]
    metric['loss'] = metrics_raw['loss'][-1]
    metric['val_accuracy'] = metrics_raw['val_accuracy'][-1]
    # save to json:
    hist_json_file = 'history.json' 
    with open(hist_json_file, mode='w') as f:
        json.dump(metrics_raw,f)

    hist_json_file = 'metrics.json' 
    with open(hist_json_file, mode='w') as f:
        json.dump(metric,f)

    model.save(best_model_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.params)