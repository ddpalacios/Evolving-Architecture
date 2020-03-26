from Setup_ import Setup
import os
# import numpy as np
# from create_population import Create_Population
import imp
# from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise




class load_file:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        hal = np.load(self.file)
        X_train, y_train, X_test, y_test,X_val, y_val = [hal[f] for f in hal.files]
        return X_train, y_train, X_test, y_test,X_val, y_val 



if __name__ == "__main__":
    # kerascodeepneat = imp.load_source("kerascodeepneat", "/home/daniel/github/Evolving-Architecture/base/kerascodeepneat.py")

    generations = 2
    training_epochs = 2
    final_model_training_epochs = 2
    population_size = 1
    blueprint_population_size = 4
    module_population_size = 4
    n_blueprint_species = 3
    n_module_species = 3
    validation_split = 0.15
    batch_size = 128


    create_dir("models/")
    create_dir("images/")

    setup = Setup()
    data_dir = 'dogs_vs_cats/train/'
    setup.dataset(data_dir)
#     compiler = setup.compiler_(lr=.001)


#     my_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
#     print(my_dataset)
#     es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=15)
#     csv_logger = CSVLogger('training.csv')
    
#     custom_fit_args = {"generator": datagen.flow(x_train, y_train, 
#                         batch_size=batch_size),
#                         "steps_per_epoch": x_train.shape[0] // batch_size,
#                         "epochs": training_epochs,
#                         "verbose": 1,
#                         "validation_data": (x_test,y_test),
#                         "callbacks": [es, csv_logger]
# }       

#     improved_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
#     improved_dataset.custom_fit_args = custom_fit_args
#     my_dataset.custom_fit_args = None

    
#     population = kerascodeepneat.Population(my_dataset, 
#                                             input_shape=x_train.shape[1:], 
#                                             population_size=population_size, 
#                                             compiler=compiler)

#     create = Create_Population(population, num_of_classes=2)

#     create.modules(module_population_size, n_module_species)
#     create.blueprints(blueprint_population_size, n_blueprint_species)
    

#     iteration = population.iterate_generations()
                                               


