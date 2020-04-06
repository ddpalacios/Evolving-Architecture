from Setup_ import Setup
import cv2
from PIL import Image
import os, itertools 
import numpy as np 
import pandas as pd 
from numpy import save 
import matplotlib.pyplot as plt 
from create_population import Create_Population
import imp
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    kerascodeepneat = imp.load_source("kerascodeepneat", "/home/daniel/github/Evolving-Architecture/base/kerascodeepneat.py")
    setup = Setup()
    generations = 2
    training_epochs = 2
    final_model_training_epochs = 2
    population_size = 100
    blueprint_population_size = 100
    module_population_size = 100
    n_blueprint_species = 3
    n_module_species = 3
    batch_size = 128

    create_dir("models/")
    create_dir("images/")

    x_train, y_train, x_test,y_test,datagen = setup.dataset("cats_dogs.npz")
    compiler = setup.compiler_(lr=.001)

    my_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])

    my_dataset.SAMPLE_SIZE = 10000
    my_dataset.TEST_SAMPLE_SIZE = 1000

    es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=15)
    csv_logger = CSVLogger('training.csv')
    
    custom_fit_args = {"generator": datagen.flow(x_train, y_train) , 
                        "batch_size": batch_size,
                        "steps_per_epoch": x_train.shape[0] // batch_size,
                        "epochs": training_epochs,
                        "verbose": 1,
                        "validation_data": (x_test,y_test),
                        "callbacks": [es, csv_logger]
}       

    improved_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
    improved_dataset.custom_fit_args = custom_fit_args
    my_dataset.custom_fit_args = None

    population = kerascodeepneat.Population(my_dataset, 
                                            input_shape=x_train.shape[1:], 
                                            population_size=population_size, 
                                            compiler=compiler)

    create = Create_Population(population, num_of_classes=2)

    create.modules(module_population_size, n_module_species)
    create.blueprints(blueprint_population_size, n_blueprint_species)
    

    iteration = population.iterate_generations()
    
