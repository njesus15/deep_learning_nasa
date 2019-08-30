import pickle
import random

import numpy as np
from keras import Model
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

from core.cnn_model import create_trail_model
from core.dataset import DataSet
from utils.data_processing import split_sequence


class KModel():
    """
    Class implements k-nearest neighbor classifier using representational layer of trained CNN Modelread_img_io

    """

    def __init__(self, train_data, test_data, model=None, n_neighbors=5, n_steps=10, baseline=None):
        """ Init method for KModel class.

        :param train_data: Numpy array (img, label) with image data and one-hot encoding vector OR list containing
                            images paths and class used for training k-nearest neighbor fit
        :param train_data: Numpy array (img, label) with image data and one-hot encoding vector OR list containing
                            images paths class names used for testing k-nearest neighbor fits
        :param model: Trained Keras Sequential model. Required if train_data and test_data of type list
        :param n_neighbors: (int) Number of neighbors parameter for knn fit
        :param n_steps: (int) Incremental steps used in KNN fit
        :param baseline: (str list) tuples containing img paths and class names used for initial fit (optional)

        """

        self.rep_layer = model
        self.train_data = train_data
        self.test_data = test_data
        self.steps = n_steps
        self.baseline = baseline
        self.n_neighbors = n_neighbors
        self.neighbors = n_neighbors
        self.losses = None
        self.fits = None

    @property
    def neighbors(self):
        return self.__neighbors

    @neighbors.setter
    def neighbors(self, neighbor_list):
        self.__neighbors = neighbor_list

    @property
    def losses(self):
        return self.__losses

    @losses.setter
    def losses(self, losses_list):
        self.__losses = losses_list

    @property
    def fits(self):
        return self.__fits

    @fits.setter
    def fits(self, fits_list):
        self.__fits = fits_list

    def train_full_model(self):
        """ Creates a k-nearest neighbor fit for increasing increments of
            self.train_data using steps values to determine the batch size of each increment.
            If using a baseline, it will be appended to training batch at each incremenent.

            Sets the attributes self.fits and self.accuracy list of KNN classifier and floats, respectively.

            :returns np.array of final batch labels to determine nearest neighbors

            """
        global batch_x, batch_y
        baseline_x, baseline_y = [], []

        # check data type of test and train data
        if isinstance(self.test_data[0], list) and isinstance(self.train_data[1], list):
            if self.rep_layer is None:
                raise TypeError('Data is of type list but self.rep_layer is None')
            test_x, test_y = rep_layer_ouptut(self.test_data, self.rep_layer)
            train_x, train_y = rep_layer_ouptut(self.train_data, self.rep_layer)

        else:
            train_x, train_y = self.train_data[0], self.train_data[1]
            test_x, test_y = self.test_data[0], self.test_data[1]

        size = len(train_x)
        batch_size = size // self.steps
        remainder = size - batch_size * self.steps
        losses = []
        fits = []
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        if self.baseline is not None:  # check if using a baseline/reference
            print("Training on baseline")
            if len(self.baseline) <= 2:
                baseline_x, baseline_y = self.baseline[0], self.baseline[1]
            else:
                baseline_x, baseline_y = rep_layer_ouptut(self.baseline, self.rep_layer)

            clf.fit(baseline_x, baseline_y)
            losses.append(clf.score(test_x, test_y))
            print(losses)
            # fits.append(clf)

        for i in range(self.steps):
            print("testing on batch: ", i)

            # incrementally append batches of data to training batch
            clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            batch_x = train_x[0: (i + 1) * batch_size + remainder * (i // (self.steps - 1))]
            batch_y = train_y[0: (i + 1) * batch_size + remainder * (i // (self.steps - 1))]

            if self.baseline is not None:
                batch_x = np.concatenate((baseline_x, batch_x), axis=0)
                batch_y = np.concatenate((baseline_y, batch_y), axis=0)
                print(batch_x.shape)

            clf.fit(batch_x, batch_y)
            acc = clf.score(test_x, test_y)
            print(acc)
            losses.append(acc)
            fits.append(clf)

        self.losses = losses
        self.fits = fits

        return [batch_x, batch_y]

    def get_neighbors(self, full_labels, fit_id=None):
        """ Computes k neighbors (self.n_neighbor) of self.test_data at fit number fit_id.
            If fit_id is None, the neighbors for the test data are computed at every fit_id.

            Attribute self.neighbors are set using output of helper method compute_neighbors()

        :param full_labels: n.darray (labels) Final training batch labels
        :param fit_id: (int) fit index of self.fits to compute neighbors
        """
        test_x = self.test_data[0]
        test_y = self.test_data[1]
        layer_neighbor = {}
        if self.fits is None or not isinstance(self.fits, list):
            raise TypeError('The KNN model has not been trained')

        elif fit_id is not None:
            print("Getting neigbors on fit: ", fit_id)
            num_fits = len(self.fits)
            if fit_id < 0 or fit_id > num_fits:
                raise ValueError('Invalid fit_id value')

            # get neighbors of each training point using fit fit_id
            k_fit = self.fits[fit_id]
            layer_neighbor['fit_id_' + str(fit_id)] = compute_neighbors([test_x, test_y],
                                                                        full_labels,
                                                                        self.n_neighbors,
                                                                        k_fit)
            self.neighbors = layer_neighbor

        else:
            for fit_num, k_fit in enumerate(self.fits):
                layer_neighbor['fit_id_' + str(fit_num)] = compute_neighbors([test_x, test_y],
                                                                             full_labels,
                                                                             self.n_neighbors,
                                                                             k_fit)
            self.neighbors = layer_neighbor


def compute_neighbors(output_data, full_labels, k, knn_fit):
    """
    Helper method to compute the k - nearest neighbor of a k-nearnest neighbor fit

    :param output_data: list (len=2) with array data of layer outputs
    :param full_labels: (ndarray) Final training batch of labels
    :param k: k-neighbors closest to training data
    :param knn_fit: knn fit

    :return: (dictionary) neighbor_dict --> keys: str of test data indices, values: dictionary 'tmp'
                            tmp_dict --> keys: 'img_label', values: class label of image
                                               'neighbor_class', values: str list of class label neighbors
                                               'neighbor_array', values: int list of neighbor indices

    TODO: convert values of 'img_label' into string class labels. Currently are array labels
    """

    data_x, data_y = output_data[0], output_data[1]
    neighbor_dict = {}
    neighbors_dist, neighbors_ind = knn_fit.kneighbors(data_x, k, return_distance=True)

    print('Accuracy:', knn_fit.score(data_x, data_y))

    if not len(data_x) == neighbors_ind.shape[0]:
        raise ValueError('Check shapes of data frame and neighbors ind')

    for index, (neighbor, img) in enumerate(zip(neighbors_ind, data_y)):
        class_list_index = [full_labels[i].tolist() for i in
                            neighbor]  # get class name from data_frame using neighbor ind
        neighbor_dict[str(index)] = {'img_label': img,
                                     'neighbor_class': class_list_index,
                                     'neighbor_array': neighbor}
    return neighbor_dict


def save_model_outputs(train_df, test_df, id, rep_layer):
    train_x, train_y = rep_layer_ouptut(train_df, rep_layer)
    test_x, test_y = rep_layer_ouptut(test_df, rep_layer)

    path = 'vgg16_output_' + str(id) + '.npz'
    np.savez(path, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    return None


def imagenet_knn():
    """
    Equivalent to knn_increase_data but instead uses the VGG16 pretrained model.
    Images are resized and saved offline to (224, 224, 3) to fit the input layer
    of VGG16

    :return: (List) test_accuracy, (ndarray) training_set, test_set, baseline_set

    """
    from keras.applications.vgg16 import VGG16
    import numpy as np

    vgg16_model = VGG16(weights=None, include_top=False)
    vgg16_layer_id = 18
    vgg16_rep_layer = Model(inputs=vgg16_model.input,
                            outputs=vgg16_model.get_layer(index=vgg16_layer_id).output)
    reset_weights(vgg16_rep_layer)

    print(vgg16_rep_layer.summary())

    # Get the dataframe
    # data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
    #                   train_subsets=[],
    #                   test_subsets=['/001/'],
    #                   loc='local')

    arr = np.load('vgg16_tests/npz_data/vgg16_output_top_off_pooling_resized.npz')
    x1, y1, x2, y2 = arr['x1'], arr['y1'], arr['x2'], arr['y2']
    x1, y1 = shuffle(x1, y1)

    # x_test, y_test = data['test_x'], data['test_y']
    x_test, y_test = x1[-2000:], y1[-2000:]
    x_train, y_train = x1[:-2000], y1[:-2000]

    # Shuffle Data and seperate
    # train_df = data_001.test_set[2]
    # random.shuffle(train_df)
    # test_df = train_df[-2000:]
    # train_df = train_df[:-2000]

    base_x, base_y = shuffle(x2, y2)

    train_data = [x_train, y_train]
    test_data = [x_test, y_test]

    knn_model = KModel(train_data, test_data, baseline=None)
    _, neighbor_labels = knn_model.train_full_model()
    return knn_model


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def test_knn():
    # file_path = 'trail_forest_results/results/trained_models/deepnn_subset_full_exc_001.h5'
    model = create_trail_model()
    # model = load_model(file_path)
    reset_weights(model)

    rep_layer = Model(inputs=model.input,
                      outputs=model.get_layer(index=7).output)
    print(rep_layer.summary())

    data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                       train_subsets=[],
                       test_subsets=['/001/'],
                       )
    data2 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                    train_subsets=[],
                    test_subsets=['/002/'],
                    )

    # Shuffle Data
    train_df = data_001.test_set[2]
    random.shuffle(train_df)
    train_df, test_df = split_sequence(train_df, 90, 30)

    baseline = data2.test_set[2]

    knn_model = KModel(train_df, test_df, model=rep_layer, baseline=None)
    knn_model.train_full_model()

    return knn_model


def plot_results():
    no_baseline = [
        'accuracy.pickle',
        'random_nn_traintest_001_out_8_8_32.pickle',
        'random_nn_traintest_001_out_4_4_32.pickle',
        'random_nn_traintest_001_out_2_2_32.pickle',
        'random_nn_traintest_001_out_1_1_32.pickle',
        'random_nn_traintest_001_out_1_4096.pickle'
    ]

    baseline = [
        'random_nn_traintest_001_bl002_out_8_8_32.pickle',
        'random_nn_traintest_001_bl002_out_4_4_32.pickle',
        'random_nn_traintest_001_bl002_out_2_2_32.pickle',
        'random_nn_traintest_001_bl002_out_1_1_32.pickle',
        'random_nn_traintest_001_bl002_out_1_4096.pickle'
    ]

    no_baseline = ['knn_tests_rand_nn/' + sub for sub in no_baseline]
    baseline = ['knn_tests_rand_nn/' + sub for sub in baseline]

    no_bl_acc = []
    bl_acc = []

    for acc in no_baseline:
        with open(acc, 'rb') as handle:
            no_bl_acc.append(pickle.load(handle))

    for acc in baseline:
        with open(acc, 'rb') as handle:
            bl_acc.append(pickle.load(handle))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    print(len(no_bl_acc))
    for acclist in no_bl_acc:
        ax.plot(np.arange(10) + 1, acclist)
    plt.legend(['vgg16 layer 4,4,32', 'layer 8,8,32', 'layer 4,4,32', 'layer 2,2,32', 'layer 1,1,32', 'dim 4096'])
    plt.title('KNN Performance on Random NN no Baseline')
    plt.xlabel('Percent of 001 training data')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1)

    fig, ax = plt.subplots(1, 1)
    for acclist in bl_acc:
        ax.plot(acclist)
    plt.legend(['layer 8,8,32', 'layer 4,4,32', 'layer 2,2,32', 'layer 1,1,32', 'dim 4096'])
    plt.title('KNN Performance on Random NN With Baseline')
    plt.xlabel('Percent of 001 training data')
    plt.ylabel('Accuracy')
    plt.ylim(0.2, 1)
