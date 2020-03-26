
import keras

class Create_Population:
    def __init__(self, CodeepNeatPop, num_of_classes=None):
        self.population = CodeepNeatPop
        self.global_configs = {
            "module_range" : ([1, 3], 'int'),
            "component_range" : ([1, 3], 'int')
        }
        self.input_configs = {
        "module_range" : ([1, 1], 'int'),
        "component_range" : ([1, 1], 'int')
        }
        self.output_configs = {
            "module_range" : ([1, 1], 'int'),
            "component_range" : ([1, 1], 'int')
        }

        self.possible_components = {
            "conv2d": (keras.layers.Conv2D, {"filters": ([16,48], 'int'), "kernel_size": ([1, 3, 5], 'list'), "strides": ([1], 'list'), "data_format": (['channels_last'], 'list'), "padding": (['same'], 'list'), "activation": (["relu"], 'list')}),
            #"dense": (keras.layers.Dense, {"units": ([8, 48], 'int')})
        }
        self.possible_inputs = {
            "conv2d": (keras.layers.Conv2D, {"filters": ([16,64], 'int'), "kernel_size": ([1], 'list'), "activation": (["relu"], 'list')})
        }
        self.possible_outputs = {
            "dense": (keras.layers.Dense, {"units": ([32,256], 'int'), "activation": (["relu"], 'list')})
        }

        self.possible_complementary_components = {
            #"maxpooling2d": (keras.layers.MaxPooling2D, {"pool_size": ([2], 'list')}),
            "dropout": (keras.layers.Dropout, {"rate": ([0, 0.5], 'float')})
        }
        self.possible_complementary_inputs = None
        self.possible_complementary_outputs = {
            "dense": (keras.layers.Dense, {"units": ([num_of_classes,num_of_classes], 'int'), "activation": (["softmax"], 'list')})
        }


    def modules(self, module_population_size, n_module_species):
        # Start with random modules
        self.population.create_module_population(module_population_size, 
                                                self.global_configs, 
                                                self.possible_components, 
                                                self.possible_complementary_components)

        self.population.create_module_species(n_module_species)

     

    def blueprints(self, blueprint_population_size, n_blueprint_species):
        self.population.create_blueprint_population(blueprint_population_size,
                                                self.global_configs, self.possible_components, self.possible_complementary_components,
                                                self.input_configs, self.possible_inputs, self.possible_complementary_inputs,
                                                self.output_configs, self.possible_outputs, self.possible_complementary_outputs)

        self.population.create_blueprint_species(n_blueprint_species)

