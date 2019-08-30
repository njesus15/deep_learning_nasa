import random
from keras import backend as K
from keras import Model
from keras.engine.saving import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from tensorflow.python.lib.io import file_io
from trainer.cnn_model.train_local import create_trail_model
from trainer.cnn_model.train_local import split_sequence
from trainer.data.DataSet import DataSet
from trainer.kneighbors.KNN_Model import rep_layer_ouptut
import numpy as np


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def correct_data_indeces(prediction, test_y):
    neigh_ind = []
    for id, (arr, y) in enumerate(zip(prediction, test_y)):
        if all(arr == y):
            neigh_ind.append(id)
    return neigh_ind


def name():

    trained_model = load_model(
        '/Users/jesusnavarro/Desktop/trail_forest_results/results/trained_models/deepnn_subset_001.h5')
    random_model = create_trail_model()
    #reset_weights(random_model)

    random_rep = Model(inputs=random_model.input,
                       outputs=random_model.get_layer(index=7).output)

    trained_rep = Model(inputs=trained_model.input,
                        outputs=trained_model.get_layer(index=8).output)

    data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                       train_subsets=[],
                       test_subsets=['/001/'],
                       )
    # Shuffle Data
    train_df = data_001.test_set[2]
    random.shuffle(train_df)
    train_df, test_df = split_sequence(train_df, 90, 30)

    rand_train_x, rand_train_y = rep_layer_ouptut(train_df, random_rep)
    rand_test_x, rand_test_y = rep_layer_ouptut(test_df, random_rep)

    train_x, train_y = rep_layer_ouptut(train_df, trained_rep)
    test_x, test_y = rep_layer_ouptut(test_df, trained_rep)

    fit_random = KNeighborsClassifier(n_neighbors=5)
    fit_trained = KNeighborsClassifier(n_neighbors=5)

    fit_random.fit(rand_train_x, rand_train_y)
    fit_trained.fit(train_x, train_y)

    random_score = fit_random.score(rand_train_x, rand_train_y)
    trained_score = fit_trained.score(test_x, test_y)

    print('random score: ', random_score)
    print('trained score: ', trained_score)

    print("Random fit tested on trained output accuracy is: ", fit_random.score(test_x, test_y))
    print("Trained NN fit tested on random nn output accuracy is: ", fit_trained.score(rand_test_x, rand_test_y))

    print("Getting predictions")

    random_predict = fit_random.predict(rand_test_x)
    trained_predict = fit_trained.predict(test_x)

    print('got predicitions')

    rand_neigh_ind = correct_data_indeces(random_predict, rand_test_y)
    train_neigh_ind = correct_data_indeces(trained_predict, test_y)

    sim_neigh = []
    if len(rand_neigh_ind) > len(train_neigh_ind):
        for val in rand_neigh_ind:
            if val in train_neigh_ind:
                sim_neigh.append(val)
    else:
        for val in train_neigh_ind:
            if val in rand_neigh_ind:
                sim_neigh.append(val)

    rand_neigh_sim = [rand_test_x[i] for i in sim_neigh]
    train_neigh_sim = [test_x[i] for i in sim_neigh]

    rand_neigh_dist, _ = fit_random.kneighbors(rand_neigh_sim)
    trained_neigh_dist, _ = fit_trained.kneighbors(train_neigh_sim)
    print('done getting neighbors')

    rand_rangeL = []
    train_rangeL = []
    for rand, trained in zip(rand_neigh_dist, trained_neigh_dist):
        rand_range = np.abs(np.amax(rand) - np.amin(rand))
        train_range = np.abs(np.amax(trained) - np.amin(trained))
        rand_rangeL.append(rand_range)
        train_rangeL.append(train_range)

    print("Rand max, min, mean, std", np.amax(rand_rangeL), np.amin(rand_rangeL), np.mean(rand_rangeL),
          np.std(rand_rangeL))
    print("Train max, min, mean, std", np.amax(train_rangeL), np.amin(train_rangeL), np.mean(train_rangeL),
          np.std(train_rangeL))

if __name__ == '__main__':
        import torch.nn

        input_size = 3 * 101*101
        data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                           train_subsets=[],
                           test_subsets=['/001/'],
                           type='arr'
                           )
        datax, datay = data_001.test_set[0], data_001.test_set[1]

        datax, datay = shuffle(datax, datay)

        train_x, train_y = datax[:-2000], datay[:-2000]
        test_x, test_y = datax[-2000:], datay[-2000:]

        train_x = torch.tensor(train_x)
        train_x = train_x.view(train_x.size(0), input_size)

        test_x = torch.tensor(test_x)
        test_x = test_x.view(test_x.size(0), input_size)


        dense = torch.nn.Sequential(torch.nn.Linear(input_size, 1024),
                                    torch.nn.ReLU()).float()

        out_train = dense(train_x.float())
        out_test = dense(test_x.float())

        clf = KNeighborsClassifier()
        clf.fit(out_train.detach().numpy(), train_y)
        s = clf.score(out_test.detach().numpy(), test_y)
        print(s)

