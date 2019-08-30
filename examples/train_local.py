# Importing the Keras libraries and packages
import pickle

import keras
import numpy as np
from imageio import imread
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.models import Sequential
# Questions:
# How were the convolution parameters chosen?
from tensorflow.python.lib.io import file_io

from utils.data_processing import split_sequence


def create_trail_model(mean=0.0, std=0.1):
    k = keras.initializers.RandomNormal(mean=mean, stddev=std, seed=None)
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(101, 101, 3), activation='relu', kernel_initializer=k))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=k))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=k))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a fourth convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=k))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=k))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 4 - Full connection
    classifier.add(Dense(units=4096, activation='relu', kernel_initializer=k))
    classifier.add(Dense(units=3, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator


# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    rotation_range = 18)


########################################
def split_data_frame(data_frame, n):
    split = [[], []]

    counts = {}
    for row in data_frame:
        count = counts.get(row[1], 0)
        if count < n:
            split[0].append(row)
        else:
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
        # append_path = path[2:]
        # true_path = 'gs' + append_path

        append_path = path[24:]
        true_path = "/Users/jesusnavarro/Desktop/DataSet/" + append_path

        if state % 300.0 == 0.0:
            print("loading " + data_type + " data: " + str(state / len(paths_df)) + '% ', true_path)

        if subset_data in true_path and "tiny" in true_path:
            with file_io.FileIO(true_path, 'rb') as f:
                img = f.read()
                img = imread(img)
                if data_type == 'test':
                    img = img * (1.0 / 255.0)
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


def test_train_deep_nn(datasets=None, train_file='gs://dataset-jesus-bucket/DataSet/',
                       job_dir='gs://dataset-jesus-bucket/', **args):
    # create model
    classifier = create_trail_model()

    # Read the data containing gs paths
    file_stream = file_io.FileIO("gs://data-daisy/full_gs_paths_subset3.pickle", mode='rb')
    data_frame = pickle.load(file_stream)

    # Will split the training due to memory consumption

    if datasets == None:
        datasets = ['/000/', '/001/', '/002/', '/003/', '/004/', '/006/', '/005/', '/007/', '/008/', '/010/', '/009/',
                    '/011/']

    split = split_sequence(data_frame, 60, 15)

    # get numpy arrays

    for subset_data in datasets:

        classifier = create_trail_model()

        subset_path = "deepnn_subset_" + subset_data[1:4] + ".h5"

        train_x, train_y, normalized_check = read_data_file_io(split[0], subset_data, data_type="train")
        test_x, test_y, normalized_check = read_data_file_io(split[1], subset_data, data_type="train")

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

        # train_set = train_datagen.flow_from_dataframe(train_df,
        #                                           target_size=(101, 101),
        #                                           batch_size=32,
        #                                           class_mode='categorical')
        # test_set = test_datagen.flow_from_dataframe(test_df,
        #                                          target_size=(101, 101),
        #                                          batch_size=32,
        #                                          class_mode='categorical')

        # Augements the data. Note that we the previous version uses flow_from_dataframe
        # however since this is being run on Google Cloud, the data is first imported into arrays
        # therefore train_datagen.flow() is used.

        train_set = train_datagen.flow(train_x, train_y, batch_size=32, shuffle=True)
        test_set = test_datagen.flow(test_x, test_y, batch_size=32, shuffle=True)

        classifier.fit_generator(train_set,
                                 steps_per_epoch=train_x.shape[0] // 32,
                                 epochs=5,
                                 validation_data=test_set,
                                 validation_steps=test_x.shape[0] // 32)

        print("Done training dataset subset: " + subset_path)

        model_path = '/Users/jesusnavarro/Desktop/deepnn_002.h5'
        classifier.save(model_path)

    # with file_io.FileIO(model_path, mode='rb') as f:
    # with file_io.FileIO(os.path.join('gs://dataset-bucket-jesus/', model_path), mode='wb+') as of:
    # of.write(f.read())
    # print('saved')

    # classifier.save("deepnn_trail_60_15_ds2short.h5")

    # Save data to google cloud storage: - subset1: files 000 and 003 are trained
    #                                    - subset2: All files are trained

# path = "/Users/jesusnavarro/Desktop/trail_project/"
# test_train_deep_nn(path)
# test_load_model()


# test_train_deep_nn(**arguments)
# test_load_model(**arguments)
# print("end")

# model = create_trail_model()
