import numpy as np


def plot_results():
    import pickle
    losses = []
    root = '/Users/jesusnavarro/Desktop/knn_tests_rand_nn/'
    paths = ['random_nn_traintest_001_bl002_out_1_4096.pickle',
             'random_nn_traintest_001_out_1_4096.pickle']

    paths = [root + path for path in paths]

    for path in paths:
        print('loading ', path)
        with open(path, 'rb') as handle:
            losses.append(pickle.load(handle))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    for loss in losses:
        print(loss)
        if len(loss) == 10:
            ax.plot((np.arange(10) + 1) * 10, loss, marker='*')
        else:
            ax.plot(np.arange(11) * 10, loss, marker='*')
    plt.title('Random NN KNN Performance: Output dim 4096')
    plt.xlabel('Percent of Dataset 1')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(['With Baseline 002',
                'No Baseline'])

    plt.show()

# plot_results()
