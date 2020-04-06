
import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir
import cv2
import numpy as np
from numpy import save

class load_file:
    def __init__(self, file):
        self.file = file

    def get_data(self):
        hal = np.load(self.file)
        X_train, y_train, X_test, y_test = [hal[f] for f in hal.files]
        return X_train, y_train, X_test, y_test 


class Setup:

    def to_categorical(y, nb_classes=None):
      
        y = np.asarray(y, dtype='int32')
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    

    def dataset(self, data):
        x_train, y_train, x_test, y_test = load_file(data).get_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        
        y_train= to_categorical(y_train, num_classes=len(np.unique(y_train)))
        y_test= to_categorical(y_test, num_classes=len(np.unique(y_test)))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        validation_split = 0.15
        #training
        batch_size = 32
      
        print(y_train.shape, y_test.shape)

        datagen = self._augment(x_train)

        return x_train, y_train, x_test, y_test, datagen


    def compiler_(self, loss= "categorical_crossentropy", optimizer= "keras.optimizers.Adam", lr=0.001, metrics = "accuracy"):
        
        optimizer = "{}({})".format(optimizer, lr)
        compiler = {"loss": loss, "optimizer":optimizer, "metrics":[metrics]}

        return compiler



    def _augment(self, data):
         # #data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            )

        datagen.fit(data)
        return datagen
