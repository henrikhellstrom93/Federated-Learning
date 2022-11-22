#Dependencies
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

#Global variables
layers_with_weights = ["conv2d", "batch", "dense"]

#K = number of devices
#local_datasets = list with K entries where each entry is a tuple with 2 entires: train_image, train_label
#global_dataset = tuple with 4 entires: train_image, train_label, test_image, test_label
def load_dataset(K):
    #Dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Converting the pixels data to float type
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # Standardizing
    train_images = train_images / 255
    test_images = test_images / 255 

    # One hot encoding the target class (labels)
    num_classes = 10
    train_labels = utils.to_categorical(train_labels, num_classes)
    test_labels = utils.to_categorical(test_labels, num_classes)
    global_dataset = (train_images, train_labels, test_images, test_labels)
    
    #Split global dataset into local datasets for each device
    local_datasets = __split_dataset(K, global_dataset)
    return local_datasets, global_dataset

def __split_dataset(K, dataset):
    N_train = dataset[0].shape[0]
    N_test = dataset[1].shape[0]
    ret = []
    
    for k in range(K):
        #Train data
        start = k*int(N_train/K)
        end = int(N_train/K) + k*int(N_train/K)
        train_x_k = dataset[0][start:end]
        train_y_k = dataset[1][start:end]
        ret.append((train_x_k, train_y_k))
        
    return ret

#Model setup
def setup_models(K, num_classes, batch_norm, dropout):
    model_list = []
    for k in range(K):
        model_list.append(__initialize_model(num_classes, batch_norm, dropout))
    global_model = __initialize_model(num_classes, batch_norm, dropout)
    return model_list, global_model

def __initialize_model(num_classes, batch_norm, dropout):
    #The weights of the layers are initialized when they are added to the Sequential object
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    if dropout == True:
        model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    if dropout == True:
        model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    if dropout == True:
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    if batch_norm == True:
        model.add(layers.BatchNormalization())
    if dropout == True:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

#Trains all models in model_list
def local_training(model_list, datasets, ep, steps):
    for k, model in enumerate(model_list):
        train_images = datasets[k][0]
        train_labels = datasets[k][1]
        history = model.fit(train_images, train_labels, batch_size=64, epochs=ep, steps_per_epoch=steps, verbose=0)
    return model_list
        
#Sets weights of global_model to the arithmetic mean of all model weights in model_list
def global_update(model_list, global_model):
    K = len(model_list)
    num_layers = len(global_model.layers)

    for l in range(num_layers):
        layer_name = global_model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #Depending on the layer, there will be different kinds of weights. 
            #Some layers have only weights+biases, in which case len(global_model.layers[l].weights)=2.
            #Some layers also have configurable parameters, such as the BatchNormalization layer
            #which has 2 extra parameters (gamma and beta) leading to len(global_model.layers[l].weights)=4.
            weight_list_len = len(global_model.layers[l].weights)
            for w_idx in range(weight_list_len):
                new_layer = tf.zeros(global_model.layers[l].weights[w_idx].get_shape())
                for model in model_list:
                    new_layer = new_layer + model.layers[l].weights[w_idx]/K
                global_model.layers[l].weights[w_idx].assign(new_layer)

    return model_list, global_model

#Sets the weights of all models in model_list to equal those of global_model
def model_broadcast(model_list, global_model):
    num_layers = len(global_model.layers)

    for l in range(num_layers):
        layer_name = global_model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #See global_update() for explanation to weight_list_len
            weight_list_len = len(global_model.layers[l].weights)
            for w_idx in range(weight_list_len):
                global_weights = global_model.layers[l].weights[w_idx]
                for model in model_list:
                    model.layers[l].weights[w_idx].assign(global_weights)
                
    return model_list, global_model