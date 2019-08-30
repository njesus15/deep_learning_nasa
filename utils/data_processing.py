import numpy as np
from imageio import imread
from tensorflow.python.lib.io import file_io


def split_sequence(data_frame, n, buffer):
    """ Partitions data_frame into test and train and uses a buffer to
        avoid extracting images too similar (sequential data)

        :param data_frame: list containing path names
        :param n: int of test data appended for each interval of 2 * n iterations
        :param buffer: int number of images not selected per 2 * n intervals

        :returns list with test and training data paths frames
    """
    split = [[], []]

    counts = {}
    for row in data_frame:
        count = counts.get(row[1], 0)
        if count % (2 * n) < n:
            split[0].append(row)
        elif count % n > buffer and count % n < n - buffer:
            #  test data
            split[1].append(row)
        counts[row[1]] = count + 1
    return split


def read_img_io(file_path_df):
    """
    Helper method to read data into array

    :param file_path_df: data frame containing file name paths and class lables
    :return: numpy array of image representation and numpy array of labels

    """
    key = {'right': np.array([1, 0, 0]),
           'center': np.array([0, 1, 0]),
           'left': np.array([0, 0, 1])}

    if not isinstance(file_path_df, list):  # Nest as a list if itsa singleton
        file_path = [file_path_df]
    else:
        file_path = file_path_df

    labels = []
    imgs = []

    for paths in file_path:
        if len(paths) == 2:  # images, labels
            file_name = paths[0]
            class_ = key[paths[1]]
            labels.append(class_)
        else:
            file_name = paths  # dataframe only contains path string

        with file_io.FileIO(file_name, 'rb') as f:
            img = f.read()
            img = imread(img)
            img = img * (1.0 / 255.0)
            imgs.append(img)

    return np.array(imgs), np.array(labels)


def rep_layer_ouptut(paths, model, logs=True):
    """
    Computes the prediction of the neural network 'model'.
    In this application, the prediction of 'model' is the representational layer and not the softmax activation.

    :param paths: List of tuples containing (img_path_names, class label)
    :param model: Keras model without output layer

    :return: (ndarray) images, (int) number of images
    """
    output = []
    size = len(paths)
    i = 1
    labels = []
    KEY = {'right': np.array([1, 0, 0]),
           'center': np.array([0, 1, 0]),
           'left': np.array([0, 0, 1])}

    for file in paths:
        img_path = file[0]
        label = file[1]
        labels.append(KEY[label])
        # get img array
        img_arr, _ = read_img_io(img_path)
        int_out = model.predict(img_arr)
        output.append(int_out)
        if size > 100 and logs:
            if i % float((size // 100)) == 0.0:  # logs of loaded data percentage
                print("Output % {:g} of the data".format(float('{:.4g}'.format(100.0 * float(i) / float(size)))))
            i += 1
    return np.array(output).reshape(size, -1), np.array(labels)
