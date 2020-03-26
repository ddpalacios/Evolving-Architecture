from os import listdir
import cv2
import os
import numpy as np
from numpy import save

def read_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        return None


def prep_data(images):
    m = len(images)
    n_x = ROWS*COLS*CHANNELS
    X = np.ndarray((n_x,m), dtype=np.uint8)
    y = np.zeros((1,m))
    print("X.shape is {}".format(X.shape))
    for i,image_file in enumerate(images) :
        image = read_image(image_file)
        if image is None:
            continue
        X[:,i] = np.squeeze(image.reshape((n_x,1)))
        if 'dog' in image_file.lower() :
            y[0,i] = 1
        elif 'cat' in image_file.lower() :
              y[0,i] = 0
        else : # for test data
              y[0,i] = image_file.split('/')[-1].split('.')[0]

        if i%5000 == 0 :
              print("Proceed {} of {}".format(i, m))

    return X,y


def reshape_images(X, y):
    ROWS = 28
    COLS = 28
    data = []
    labels = []
    CHANNELS = 3
    for each_image in range(X.shape[0]):
        image = X[each_image].reshape((ROWS,COLS, CHANNELS))
        data.append(image)
    
    for each_image in range(X.shape[0]):
        labels.append(y[each_image,0])
    return np.array(data), np.array(labels)




if __name__ == "__main__":
    TRAIN_DIR = '/home/daniel/github/Evolving-Architecture/dataset_kaggle/train/'
    TEST_DIR = '/home/daniel/github/Evolving-Architecture/dataset_kaggle/test1/'
    ROWS = 28
    COLS = 28
    CHANNELS = 3

    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
    test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

    X_train, y_train = prep_data(train_images)
    X_test, test_idx = prep_data(test_images)
    x_train, y_train = reshape_images(X_train.T, y_train.T)
    x_test, y_test = reshape_images(X_test.T, test_idx.T)


    print("Saving...",X_test.T[0], test_idx[0], X_train.shape)
    np.savez_compressed('cats_dogs.npz',
                    X_train=x_train,
                    y_train=y_train,
                    X_test=x_test,
                    y_test=y_test)
