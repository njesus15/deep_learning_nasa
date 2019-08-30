# Importing the Keras libraries and packages
import os
import pickle
import random

import keras
import numpy as np
from keras import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier

from core.dataset import DataSet


def create_trail_model():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(101, 101, 3), activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a fourth convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units=200, activation='relu'))
    classifier.add(Dense(units=3, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


def save_weights():
    dir = '/Users/jesusnavarro/Desktop/trail_project/results/trained_models/'

    saved_models = os.listdir(dir)

    for model in saved_models:
        print(model)
        if '.h5' in model:
            tmp_model = load_model(dir + model)
            path_to_save = '/Users/jesusnavarro/Desktop/trail_project/results/' + 'saved_weights/' + model
            tmp_model.save_weights(path_to_save)


def load_from_url():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    PATH = 'https://github.com/njesus15/forest_trail_deep_learning/blob/master/results/saved_weights/deepnn_subset_full_exc_001_002.h5?raw=true'
    model = create_trail_model()

    weights_path = keras.utils.get_file('deepnn_subset_001.h5', PATH)
    weights = model.load_weights(weights_path)

    return model


def knn_accuracy(clf, x_train, y_train, model):
    int_output = model.predict(x_train)
    int_output = int_output.reshape(x_train.shape[0], -1)
    prediction = clf.predict(int_output)
    err = 0
    for pred, y in zip(prediction, y_train):
        if any(pred - y):
            err += 1
    acc = ((y_train.shape[0] - err) / y_train.shape[0])
    print('Knn acc is', acc)
    return acc


def knn_increase_data():
    model = load_from_url()

    rep_layer = Model(inputs=model.input,
                      outputs=model.get_layer(index=7).output)
    data_002 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                       train_subsets=[],
                       test_subsets=['/002/'],
                       type='nl',
                       loc='gs')

    datasets = ['/001/', '/002/', '/003/', '/004/', '/005/', '/006/', '/007/', '/008/', '/009/', '/010/', '/011/']
    accuracy_list = {}

    for ds in datasets:
        print(ds)
        data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                           train_subsets=[],
                           test_subsets=[ds],
                           type='nl',
                           loc='gs')

        accuracy_list[ds] = []

        data_array = data_001.test_set
        d2 = data_002.test_set

        list_to_randomize = []
        list_test = []

        for (x, y) in zip(data_array[0], data_array[1]):
            list_to_randomize.append([x, y])

        random.shuffle(list_to_randomize)
        n = 10
        batch_size = len(list_to_randomize) // n
        remainder = len(list_to_randomize) - batch_size * n
        print(batch_size)

        for (x, y) in zip(d2[0], d2[1]):
            list_to_randomize.append([x, y])

        # extract data to test (001 dataset up to batch size * n + remainder
        x_001_randarr = np.array([item[0] for item in list_to_randomize[0: n * batch_size + remainder - 1]])
        y_001_randarr = np.array([item[1] for item in list_to_randomize[0: n * batch_size + remainder - 1]])

        x_002_list = [item[0] for item in list_to_randomize[n * batch_size + remainder:]]  # used for ref. point
        y_002_list = [item[1] for item in list_to_randomize[n * batch_size + remainder:]]

        clf = KNeighborsClassifier()  # creat KNN object

        # train with dataset 2
        x_002_arr = np.array(x_002_list)
        int_output = rep_layer.predict(x_002_arr)
        int_output = int_output.reshape(x_002_arr.shape[0], -1)
        clf.fit(int_output, np.array(y_002_list))

        init_loss = knn_accuracy(clf, x_001_randarr, y_001_randarr, rep_layer)  # Test on 001
        accuracy_list[ds].append(init_loss)
        z = 1

        for i in range(10):
            print("Fitting on batch number:", z)

            x_test_list = [item[0] for item in list_to_randomize[0:(i + 1) * batch_size - 1 + remainder * (i // 9)]] + [
                item[0] for item in list_to_randomize[n * batch_size + remainder:]]
            y_test_list = [item[1] for item in list_to_randomize[0:(i + 1) * batch_size - 1 + remainder * (i // 9)]] + [
                item[1] for item in list_to_randomize[n * batch_size + remainder:]]

            x = np.array(x_test_list)
            y = np.array(y_test_list)

            print(x.shape, y.shape)

            int_output = rep_layer.predict(x)
            int_output = int_output.reshape(x.shape[0], -1)

            clf.fit(int_output, y)

            accuracy = knn_accuracy(clf, x_001_randarr, y_001_randarr, rep_layer)
            print(accuracy)
            accuracy_list[ds].append(accuracy)

            z += 1

    with open('gs://data-daisy/knn_test.pickle', 'wb+') as handle:
        pickle.dump(accuracy_list, handle)
    return accuracy_list


knn_increase_data()
