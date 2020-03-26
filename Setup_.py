
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir
import cv2
import numpy as np
from numpy import save
class Setup:

    def dataset(self, data):
        data = []
        labels = []
        
        for file in listdir('/home/daniel/github/Evolving-Architecture/dogs-vs-cats/train/'):
            random_num = np.random.rand()

            if file.startswith("cat"):
                img = cv2.imread('/home/daniel/github/Evolving-Architecture/dogs-vs-cats/train/'+file)
                data.append(img)
                labels.append(0)


            elif file.startswith('dog'):
                img = cv2.imread('/home/daniel/github/Evolving-Architecture/dogs-vs-cats/train/'+file)
                data.append(img)
                labels.append(1)
            print("Data size:",len(data))
            if len(data) >= 10000:
                break
        data = np.array(data)
        labels = np.array(labels)
        save("np_data.npy",data)
        save("np_labels.npy", labels)
        # np.savez_compressed('all_data.npz',
        #             X_train=X_train,
        #             y_train=y_train,
        #             X_test=X_test,
        #             y_test=y_test,
        #             X_val = X_val,
        #             y_val = y_val)

            
        #  # The data, split between train and test sets:
        # x_train, y_train, x_test, y_test, xval,yval = data
        # print('x_train shape:', x_train.shape)
        # print(x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')
        # print("Loaded data")
        # y_train = keras.utils.to_categorical(y_train, 2)
        # y_test = keras.utils.to_categorical(y_test, 2)

        


        # datagen = self._augment(x_train)

        # return x_train,y_train,x_test,y_test, datagen


    def compiler_(self, loss= "categorical_crossentropy", optimizer= "keras.optimizers.Adam", lr=0.005, metrics = "accuracy"):
        
        optimizer = "{}({})".format(optimizer, lr)
        compiler = {"loss": loss, "optimizer":optimizer, "metrics":[metrics]}

        return compiler



    def _augment(self, data):
         # #data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            )

        datagen.fit(data)
        return datagen
