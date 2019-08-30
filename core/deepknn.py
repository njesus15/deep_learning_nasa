import pickle
import random

import numpy as np
import pandas as pd
from keras import Model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.neighbors import LSHForest
from sklearn.utils import shuffle

from core.cnn_model import create_trail_model
from core.dataset import DataSet
from utils.data_processing import read_img_io
from utils.data_processing import rep_layer_ouptut
from utils.data_processing import split_sequence


class DknnModel():
    """ Deep K-Nearest Neighbor implementation model on the Forest Trail Dataset
    """

    def __init__(self, root_path, train_set, test_set, n_test_points=10, n_neighbors=5):
        """
        :param root_path: (str) root directory of the dataset
        :param train_set: (list) image path names for training
        :param test_set: (list) image path names for testing
        :param n_test_points: (int) number of points from test_set to compute dknn
        :param n_neighbors: (int) number of neighbors to compute for each test point
        """
        self.k_neighbors = n_neighbors
        self.n_test_points = n_test_points
        self.root = root_path
        self.__test_set = test_set
        self.__train_set = train_set
        self.__validation_set = None
        self.__model = None
        self.__calibration_set = None
        self.__rep_layers = None
        self.__knn_fits = None
        self.__layer_outputs = None
        self.__calibration_ncvals = None
        self.__p_values = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_info):
        """
        Setter method for model
        TODO: If path arg exists, load data associated with that particular dknn. FIX REP_LAYER ATTRIBUTES ETC. IN LOOP.

        """
        if model_info is None:
            model = create_trail_model()
            model = self.train_model(model)
            self.__model = model

        else:  # set attributes from input data
            assert len(model_info) == 5 and isinstance(model_info, list)
            model = model_info[0]
            self.__model = model
            self.__train_set = model_info[1]
            test = model_info[2]
            self.__test_set = test[:self.n_test_points]
            self.__calibration_set = model_info[3]
            self.__validation_set = model_info[4]

        # exclude layer 9, a flatten layer
        n = len(model.layers)
        self.__rep_layers = [Model(inputs=model.input,
                                   outputs=model.get_layer(index=i).output) for i in range(n - 1) if not i == 8]
        assert len(self.__rep_layers) == n - 2

    @property
    def rep_layers(self):
        return self.__rep_layers

    @rep_layers.setter
    def rep_layers(self, layers):
        self.__rep_layers = layers

    @property
    def train_set(self):
        return self.__train_set

    @train_set.setter
    def train_set(self, data_frame):
        self.__train_set = data_frame

    @property
    def validation_set(self):
        return self.__validation_set

    @validation_set.setter
    def validation_set(self, valid_set):
        self.__validation_set = valid_set

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, data_frame):
        self.__test_set = data_frame

    @property
    def calibration_set(self):
        return self.__calibration_set

    @calibration_set.setter
    def calibration_set(self, data_frame):
        self.__calibration_set = data_frame

    @property
    def layer_outputs(self):
        return self.__layer_outputs

    @layer_outputs.setter
    def layer_outputs(self, output_data):
        print(type(output_data))
        if not isinstance(output_data, list) and not isinstance(output_data, np.ndarray):
            raise ValueError('Input must be a list containing outputs and class labels')
        self.__layer_outputs = output_data

    @property
    def knn_fits(self):
        return self.__knn_fits

    @knn_fits.setter
    def knn_fits(self, fits_list):
        self.__knn_fits = fits_list

    @property
    def calibration_ncvals(self):
        return self.__calibration_ncvals

    @calibration_ncvals.setter
    def calibration_ncvals(self, values):
        self.__calibration_ncvals = values

    @property
    def p_values(self):
        return self.__p_values

    @p_values.setter
    def p_values(self, values):
        self.__p_values = values

    def train_model(self, model):
        """
        Train model and set data attributes

        The model and datasets (test, train, calibration) are saved to 'workspace.pickle'.
        Training data is randomized and training is done using .flow() where inputs are
        numpy arrays.

        """

        # Get dataframe of paths
        dataset = DataSet(self.root, self.train_set, self.test_set, type='list')
        paths_dataframe = dataset.train_set[2]
        train_df, validation_df, calibration_df = split_data(paths_dataframe)

        # set new parameters for train and calibration_set
        test_df = dataset.test_set[2]
        random.shuffle(test_df)
        self.__test_set = test_df
        self.__train_set = train_df
        self.__validation_set = validation_df
        self.__calibration_set = calibration_df

        train_images, train_labels = read_img_io(train_df)

        # randomize the data
        train_images, train_labels = shuffle(train_images, train_labels)

        validation_images, validation_labels = read_img_io(validation_df)
        train_datagen = ImageDataGenerator(rescale=1,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           rotation_range=18)

        valid_datagen = ImageDataGenerator()

        train_set = train_datagen.flow(train_images, train_labels, batch_size=32, shuffle=True)
        valid_set = valid_datagen.flow(validation_images, validation_labels, batch_size=32, shuffle=True)

        model.fit_generator(train_set,
                            steps_per_epoch=train_images.shape[0] // 32,
                            epochs=5,
                            validation_data=valid_set,
                            validation_steps=validation_images.shape[0] // 32)

        # save to current workspace
        with open('workspace.pickle', 'wb+') as handle:
            pickle.dump([model, train_df, test_df, calibration_df, validation_df], handle)
        return model

    def compute_layer_outputs(self):
        """" Construct knn fits at each layer output """

        global train_y
        if self.__rep_layers is None or self.__calibration_set is None:
            raise TypeError

        layer_output = []
        for id, model in enumerate(self.__rep_layers[1:]):  # calib_data a dataframe
            # Get output at each layer
            train_output, train_y = rep_layer_ouptut(self.__train_set, model, logs=False)
            layer_output.append(train_output)
        self.layer_outputs = [layer_output, train_y]

    def multi_sets(self, layer_id=None):

        """ Computes multisets (defined below) for each layer of the nn model

        Multiset at layer l for test_point z (M_l) is computed as follows:
        - Training data (X, Y)
        - Compute set N <-- k points in X closest to z at layer l
        - Compute M_l   <-- {Y_i: i in N}

        :return: ndarray (layer_num, data_point_num, k-neighbors) for test, calibration, and validation

        TODO: Compute and return multiset for validation data
        """
        test_multiset_l = []
        calibration_multiset_l = []

        if layer_id is not None:
            rep_layers = [self.rep_layers[layer_id]]
        else:
            rep_layers = self.rep_layers[1:]

        for id, (tmp_model, layer_output) in enumerate(zip(rep_layers, self.layer_outputs[0])):
            print("Computing multiset for layer ", id)
            train_labels = self.layer_outputs[1]

            # compute LSF fit
            k_fit = LSHForest(n_neighbors=self.k_neighbors)
            k_fit.fit(layer_output, self.layer_outputs[1])

            # Compute output of model layer
            test_x, test_y = rep_layer_ouptut(self.test_set, tmp_model, logs=False)
            calib_x, calib_y = rep_layer_ouptut(self.calibration_set, tmp_model, logs=False)

            if not test_x.shape[1] == layer_output.shape[1]:
                raise ValueError('Training data and test data are different shape', test_x.shape, layer_output.shape)

            test_dist, test_neighbors = k_fit.kneighbors(test_x, self.k_neighbors)
            calib_dist, calib_neighbors = k_fit.kneighbors(calib_x, self.k_neighbors)

            t_neighbors = []
            c_neighbors = []

            for test_neigh, calib_neigh in zip(test_neighbors, calib_neighbors):
                t_neighbors.append(np.array([train_labels[i] for i in test_neigh]))

            for calib_neigh in calib_neighbors:
                c_neighbors.append(np.array([train_labels[i] for i in calib_neigh]))

            test_multiset_l.append(np.array(t_neighbors))
            calibration_multiset_l.append(np.array(c_neighbors))

        return test_multiset_l, calibration_multiset_l

    def compute_nonconformity(self, multiset):
        """
        Computes nonconformity of multiset matrix

        :param multiset: ndarray of size (layer_num, img_num, k_neighbors)

        :returns: ndarray of size (layer_num, class_num (3), img_num)

        Nonconformity for data point x and label j is defined as:
                        [SUM at each layer l] (|i in M_l : i !=  j|)

        The nonconformity matrix is the nonconformity values of each data point and each label

        High level flow:
            1. Construct array representation of each label --> 'filter_j'
            2. Iterate through each layer ---> 'layer_ind'
            3. Iterate through each data point --> 'data_point_ind'
            4. iterate through each neighbor --> 'neighbor'
            5. Pass filter_j through each neighbor and increase tmp_val if filter_j != neighbor
            6. Assign tmp_val to matrix[layer_ind, class_j, data_point_ind]

        The ouptut is the sum across layers of matrix (axis=0)

        """

        global conform_matrix
        layer_num = len(multiset)
        test_points = multiset[0].shape[0]
        conform_matrix = np.zeros((layer_num, 3, test_points))

        for class_j in range(3):
            filter = np.zeros(3)
            filter[class_j] = 1
            # iterate through layers
            for layer_ind, layer_ms in enumerate(multiset):
                # iterate through data points
                for data_point_ind, data_point in enumerate(layer_ms):
                    tmp_val = 0
                    # iterate through k-neighbors
                    for neighbor in data_point:
                        if not all(neighbor == filter):
                            tmp_val += 1
                        # save (layer number, data point, a_j)
                    conform_matrix[layer_ind][class_j][data_point_ind] = tmp_val

        # conform_matrix shape = (num_layers, data_points, class)
        conform_matrix_sum = np.sum(conform_matrix, axis=0)

        return conform_matrix, conform_matrix_sum

    def compute_p_vals(self, test_ncvals, size_calib):
        """ Compute emperical p-value:

            p_j(test_point) = |{ a in calibration_ncval : a > nonconrmity_val(test, class_j) }| / |calibration_ncval|

        Assuming that we check every value of calibration_ncvals, then flatten array for simplicity

        TODO: Check assumption
        """

        # iterate through class and compute p value

        assert self.calibration_ncvals is not None
        size_calib = float(size_calib)
        p_matrix = np.empty_like(test_ncvals)

        calibration = self.calibration_ncvals.flatten()

        for class_ind in range(3):
            class_j = test_ncvals[class_ind]
            for j in range(len(class_j)):
                data_j_nc = float(class_j[j])
                p_val = np.sum(calibration > data_j_nc) / float(size_calib)
                assert p_val < size_calib
                p_matrix[class_ind][j] = p_val

        return p_matrix.transpose()

    def evaluate(self):
        """ Full dknn evaluation

        :returns : pandas data frame with colunmns = ['image', 'p_val', 'prediction', '', '']
        """
        self.compute_layer_outputs()
        test_ml, calib_ml = self.multi_sets()

        # set the calibration non conformity values as parameter
        self.calibration_ncvals = self.compute_nonconformity(multiset=calib_ml)[1]
        test_ncvals = self.compute_nonconformity(multiset=test_ml)[1]

        # get size of calibration non conformity values
        size_A = len(self.calibration_ncvals.flatten())
        self.p_values = self.compute_p_vals(test_ncvals, size_A)

        # p_vales.shape = (data_points, classes)

        pred_list = [np.where(arr == np.amax(arr), 1, 0) for arr in self.p_values]
        confidence_list = [x[np.argsort(x)[-2:]] for x in self.p_values]

        data_frame = [[df[0], df[1], pred.tolist(), conf] for df, pred, conf in zip(self.test_set,
                                                                                    pred_list, confidence_list)]

        df = pd.DataFrame(data_frame, columns=['image', 'class', 'prediction', 'confidence'])
        return (df)


def split_data(data_frame):
    """ Helper method  to partition input data

    :param data_frame: (list) image paths and class labels

    :return: (list of data_frames) partitioned into train, validation, and calibration data

    """
    random.shuffle(data_frame)

    # split data 3 times. First split half assigned to training set
    first_split = split_sequence(data_frame, 60, 10)
    train_split, test_split = first_split[0], first_split[1]
    second_split = split_sequence(test_split, 30, 10)
    train_df, valid_df, callib_df = train_split, second_split[1], second_split[0]
    print('Size of train, valid, and calib data:', len(train_df), len(valid_df), len(callib_df))

    return train_df, valid_df, callib_df
# if __name__ == '__main__':

#  model_info = None
#   if os.path.exists('workspace.pickle'):
#  with open('workspace.pickle', 'rb') as handle:
# model_info = pickle.load(handle)

# dknn = DknnModel('/Users/jesusnavarro/Desktop/DataSet/',
#   ['/001/'], ['/002/'], n_test_points=40)
# dknn.model = model_info

# pd = dknn.evaluate()

# dknn.calibrate_layers()
# test_ml, calib_ml, test_label, calib_label = dknn.multi_sets()

# set the calibration non conformity values as parameter
# dknn.calibration_ncvals = dknn.compute_nonconformity(multiset=calib_ml)[1]
# test_ncvals = dknn.compute_nonconformity(multiset=test_ml)[1]

# get size of calibration non conformity values
# size_A = len(dknn.calibration_ncvals.flatten())
# dknn.p_values = dknn.compute_p_vals(test_ncvals, size_A)
