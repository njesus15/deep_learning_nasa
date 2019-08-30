# Importing the Keras libraries and packages
import argparse as ag
import pickle
import os
import random

import pickle
import random

from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from imageio import imread
from keras import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
# Questions:
# How were the convolution parameters chosen?
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import file_io


def create_trail_model():
    """ Creates and returns the Convolutional Neural Network Model
    """

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

def split_sequence(data_frame, n, buffer):

    """ Splits data frame containing paths and class labels into test and training set with a buffer to discard
        closely identical data (note the data is temporal).

    :param data_frame (list str): [[image path, class label]..]
    :param n (int) : porportional to training set
    :param buffer (int): Size discared data which goes by counts
    :return split (tuple) : ([train data_frame], [test data_frame])
    """

    split = [[], []]
    counts = {}
    for row in data_frame:
        count = counts.get(row[1], 0)
        if count % (2 * n) < n:
            split[0].append(row)
        elif count % n > buffer and count % n < n - buffer:
            # Leave buffer for test data
            split[1].append(row)
        counts[row[1]] = count + 1
    return split


def check_scaling(array_images):
    for image in array_images:
        for rgb_pixel in image:
            if np.max(rgb_pixel) > 1.0:
                print("The data is not normalized")
                return False

    print("The data is normalized")
    return True


## NOTE: BUG HERE. PATHS SHOULD START WITH GS AND NOT GC. FIX WILL BE HARDCODED IN THE INTEREST OF TIME. DEBUG LATER!

def read_data_file_io(path, subset_data, data_type):
    print("Loading dataset:")
    paths_df = path
    images = []
    labels = []
    state = 1.0

    key = {'right': np.array([1, 0, 0]),
           'center': np.array([0, 1, 0]),
           'left': np.array([0, 0, 1])}

    # file item composed of (path, class)
    print("length of path", len(paths_df))
    for file in paths_df:
        path = file[0]

        append_path = path[2:]
        true_path = 'gs' + append_path

        if state % 300.0 == 0.0:
            print("loading " + data_type + " data: " + str(state / len(paths_df)) + '% ', true_path)

        if any(prefix in true_path for prefix in subset_data) and "tiny" in true_path:
            with file_io.FileIO(true_path, 'rb') as f:
                img = f.read()
                img = imread(img)
                if data_type == 'test':
                    img = img * (1.0/255.0)
                images.append(img)
                labels.append(key[file[1]])
            state += 1.0

    labels = np.array(labels)
    images = np.array(images)

    # checks if images have been rescaled

    normalized_check = check_scaling(images)

    return images, labels, normalized_check


def fix_path(data_frame):
    path_df = []
    for file in data_frame:
        path = file[0]
        append_string = path[2:]
        true_path = 'gs' + append_string
        path_df.append([true_path, file[1]])
    return path_df

def knn_accuracy(clf, x_train, y_train, model):
    int_output = model.predict(x_train)
    int_output = int_output.reshape(x_train.shape[0], -1)
    prediction = clf.predict(int_output)
    error = np.abs(prediction - y_train) / 2
    error_count = np.sum(error)
    acc = ((y_train.shape[0] - error_count) / y_train.shape[0])
    return acc


def test_train_deep_nn(train_file='gs://dataset-jesus-bucket/DataSet/',
                       job_dir='gs://dataset-jesus-bucket/', **args):

    # create model

    # Read the data containing gs paths
    global acc_path_to_save
    file_stream = file_io.FileIO("gs://data-daisy/full_gs_paths_subset3.pickle", mode='rb')
    data_frame = pickle.load(file_stream)

    # Will split the training due to memory consumption
    datasets = [['/000/', '/001/', '/003/', '/004/', '/006/', '/005/', '/007/', '/008/', '/010/', '/009/',
                '/011/', 'full_exc_002_2']]

    split = split_sequence(data_frame, 60, 15)

    # get numpy arrays
    classifier = create_trail_model()

    for subset_data in datasets:
        subset_path = "deepnn_subset_" + subset_data[-1] + ".h5"

        train_x, train_y, normalized_check = read_data_file_io(split[0], subset_data[:-1], data_type="train")
        test_x, test_y, normalized_check = read_data_file_io(split[1], subset_data[:-1],  data_type="train")

        if normalized_check:
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / 255.0

    # ImageDataGenerator is used to append additional data that is rescaled
    # sheared, zoomed, and rotated for test and training sets
        train_datagen = ImageDataGenerator(rescale=scale_factor,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       rotation_range=18)

        test_datagen = ImageDataGenerator(scale_factor)
        acc_path_to_save = 'gs://data-daisy/model_accuracy_' + subset_data[0][:-1] + '.pickle'

    # train_set = train_datagen.flow_from_dataframe(train_df,
    #                                           target_size=(101, 101),
    #                                           batch_size=32,
    #                                           class_mode='categorical')
    # test_set = test_datagen.flow_from_dataframe(test_df,
    #                                          target_size=(101, 101),
    #                                          batch_size=32,
    #                                          class_mode='categorical')
        # Importing the Keras libraries and packages

            # classifier.save("deepnn_trail_60_15_ds2short.h5")

            # Save data to google cloud storage: - subset1: files 000 and 003 are trained
            #                                    - subset2: All files are trained


        # Augements the data. Note that we the previous version uses flow_from_dataframe
    # however since this is being run on Google Cloud, the data is first imported into arrays
    # therefore train_datagen.flow() is used.

        train_set = train_datagen.flow(train_x, train_y, batch_size=128, shuffle=True)
        test_set = test_datagen.flow(test_x, test_y, batch_size=128, shuffle=True)

        classifier.fit_generator(train_set,
                             steps_per_epoch=train_x.shape[0] // 128,
                             epochs=4,
                             validation_data=test_set,
                             validation_steps=test_x.shape[0] // 128)

        print("Done training dataset subset: " + subset_path)

        classifier.save(subset_path)

    with file_io.FileIO(subset_path, mode='rb') as f:
        with file_io.FileIO(os.path.join('gs://data-daisy/', subset_path), mode='wb+') as of:
            of.write(f.read())
            of.close()
            print('saved')
        f.close()

    print("Now testing model")

    datasets = ['/000/', '/001/', '/002/', '/003/', '/004/', '/005/', '/006/', '/007/', '/008/', '/009/', '/010/',
                '/011/']
    acc_dict = {}
    for dataset in datasets:
        x, y, _ = read_data_file_io(data_frame, dataset, data_type="test")
        loss, accuracy = classifier.evaluate(x, y)
        acc_dict[dataset] = accuracy
    x = 0
    y = 0
    print("done testing dataset now doing knn")

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

def imagenet_knn(train_file='gs://dataset-jesus-bucket/DataSet/',
                       job_dir='gs://dataset-jesus-bucket/', **args):
    import pandas as pd
    from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from keras.preprocessing import image
    import keras.backend as K
    import numpy as np
    import sys

    file_stream = file_io.FileIO("gs://data-daisy/full_gs_paths_large_size.pickle", mode='rb')
    data_frame = pickle.load(file_stream)

    vgg16_model = VGG16(weights='imagenet', include_top=True)

    vgg16_rep_layer = Model(inputs=vgg16_model.input,
                      outputs=vgg16_model.get_layer(index=21).output)

    print(vgg16_rep_layer.summary())


    x_001, y_001, normalized_check = read_data_file_io(data_frame, ['/001/'], data_type="test")
    x_002, y_002, normalized_check = read_data_file_io(data_frame, ['/002/'], data_type="test")

    x_001_list, y_001_list = x_001.tolist(), y_001.tolist()
    x_002_list , y_002_list = x_002.tolist(), y_002.tolist()


    list_to_randomize = []
    list_test = []

    for (x, y) in zip(x_001_list, y_001_list):
        list_to_randomize.append([x,y])

    random.shuffle(list_to_randomize) # shuffle data used to train
    n = 10
    batch_size = len(list_to_randomize) // n
    remainder = len(list_to_randomize) - batch_size * n
    print(batch_size)

    for (x, y) in zip(x_002_list, y_002_list):
        list_to_randomize.append([x,y])

    # extract data to test (001 dataset up to batch size * n + remainder
    x_001_randarr = np.array([item[0] for item in list_to_randomize[0: n * batch_size + remainder - 1]])
    y_001_randarr = np.array([item[1] for item in list_to_randomize[0: n * batch_size + remainder - 1]])

    x_002_list = [item[0] for item in list_to_randomize[n * batch_size + remainder:]]  # used for ref. point
    y_002_list = [item[1] for item in list_to_randomize[n * batch_size + remainder:]]

    clf = KNeighborsClassifier()  # creat KNN object
    accuracy_list = []

    # train with dataset 2
    x_002_arr = np.array(x_002_list)
    int_output = vgg16_rep_layer.predict(x_002_arr)
    int_output = int_output.reshape(x_002_arr.shape[0], -1)
    clf.fit(int_output, np.array(y_002_list))

    init_loss = knn_accuracy(clf, x_001_randarr, y_001_randarr, vgg16_rep_layer)  # Test on 001
    accuracy_list.append(init_loss)

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

        int_output = vgg16_rep_layer.predict(x)
        int_output = int_output.reshape(x.shape[0], -1)

        clf.fit(int_output, y)

        accuracy = knn_accuracy(clf, x_001_randarr, y_001_randarr, vgg16_rep_layer)
        print(accuracy)
        accuracy_list.append(accuracy)

        z += 1

    with open('gs://data-daisy/increasing_knn_acc_vgg16.pickle', 'wb+') as handle:
        pickle.dump(accuracy_list, handle)

    print(accuracy_list)

# path = "/Users/jesusnavarro/Desktop/trail_project/"
# test_train_deep_nn(path)
# test_load_model()

def load_model_test(**args):
    model_file = file_io.FileIO('gs://data-daisy/deepnn_subset_001_and_002.h5', mode='rb')
    model = load_model('gs://data-daisy/deepnn_subset_001_and_002.h5')
    print(model.summary())

print("here")

parser = ag.ArgumentParser()

# Input ArgumentsS
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=False
)

parser.add_argument(
    '--train-file',
    help='GCS or local paths for training data',
    required=False
)

args = parser.parse_args()
arguments = args.__dict__

load_model_test(**arguments)
# test_load_model(**arguments)
print("end")


