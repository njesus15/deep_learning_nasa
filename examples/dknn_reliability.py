from keras import Model
from mpl_toolkits.mplot3d import Axes3D


from trainer.dkneighbors.Dknn import DknnModel
from trainer.kneighbors.KNN_Model import read_img_io, rep_layer_ouptut
import numpy as np
from keras.models import load_model

import os
import pickle
import matplotlib.pyplot as plt


def bucket_index(value):
    bin = np.arange(0, 1.1, 0.1)
    assert len(bin) > 1 and 0 <= value < 1
    for i in range(len(bin) - 1):
        current_val = bin[i]
        forward_val = bin[i + 1]
        if current_val <= value < forward_val:
            return i
    return None


def organize_buckets(labels, predictions, cred_val):
    assert len(labels) == len(predictions) == len(cred_val)
    buckets = {}
    for i in range(10):
        buckets[str(i)] = []
    # 10 buckets
    for id, (val, predict, label) in enumerate(zip(cred_val, predictions, labels)):
        bucket_ind = bucket_index(val)
        acc_count = np.sum(all(predict == label))
        buckets[str(bucket_ind)].append([acc_count, id])

    for key in buckets.keys():
        acc_count = [val[0] for val in buckets[key]]
        ind = [val[1] for val in buckets[key]]
        if len(acc_count) == 0:
            buckets[ key] = [0, 0]
        else:
            size = float(len(acc_count))
            accuracy = float(np.sum(acc_count)) / size
            buckets[key] = [accuracy, size, ind]
    print(buckets)
    return buckets


def dknn_reliability(model, df, dknn_df):
    test_x, test_y = read_img_io(df)

    # get softmax values, predicition labels, and prediction value (argmax of prediction output)
    softmax_values = model.predict(test_x)
    softmax_pred = [np.where(arr == np.amax(arr), 1, 0) for arr in softmax_values]
    soft_max_values = np.amax(softmax_values, axis=0)

    # get dknn prediction from dknn_df and p_values for credibility and confidence
    test_dkkn_prediction = dknn_df['prediction'].tolist()
    #val, val1 = model.evaluate(test_x[:len(test_dkkn_prediction)], test_y[:len(test_dkkn_prediction)])
    #print(val, val1)
    test_p_values = dknn_df['confidence'].tolist()
    test_cred, test_conf = [val[1] for val in test_p_values], [(1 - val[0]) for val in test_p_values]

    dknn_bucket = organize_buckets(test_y[:len(test_dkkn_prediction)], test_dkkn_prediction, test_cred)
    return dknn_bucket

def plot_buckets(bucket_dict):
    fig, ax = plt.subplots(1,1)
    acc = []
    for key in bucket_dict.keys():
        tmp_list = bucket_dict[key]
        size = tmp_list[1]
        acc.append(tmp_list[0])
    ax.bar([i for i, _ in enumerate(acc)], acc)
    ticks = ['0.0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    plt.xticks([i for i, _ in enumerate(acc)], ticks)
    plt.xlabel('Credibility')
    plt.ylabel('Accuracy')

def hello():
    model_info = None
    if os.path.exists('workspace.pickle'):
        with open('workspace.pickle', 'rb') as handle:
            model_info = pickle.load(handle)
    dknn = DknnModel('/Users/jesusnavarro/Desktop/DataSet/',
                     ['/001/'], ['/003/'], n_test_points=1000, n_neighbors=30)

    dknn.model = model_info
    pd = dknn.evaluate()
    rel = dknn_reliability(dknn.model, dknn.test_set, pd)
    plot_buckets(rel)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    if os.path.exists('workspace2.pickle'):
        with open('workspace2.pickle', 'rb') as handle:
            data = pickle.load(handle)
    dknn_model = load_model('workspace2.h5')
    pd = data[0]
    test_set = data[1]
    rel = dknn_reliability(dknn_model, test_set, pd)
    buckets_df = {}

    for key in rel.keys():
        tup = rel[key]
        if len(tup) > 2:
            buckets_df[key] = [test_set[val] for val in tup[2]]
    print(buckets_df)

    import numpy as np
    from sklearn.decomposition import PCA

    rep_layer = Model(inputs=dknn_model.input,
                      outputs=dknn_model.get_layer(index=7).output)

    pca = PCA(n_components=2)

    for val in buckets_df.keys():
        if val == str(6):
            df = buckets_df[val]
            dff = []
            #for file in df:
            #    if 'center' in file[1]:
            #        dff.append(file)
            data_x, data_y = rep_layer_ouptut(df, rep_layer)
            pca.fit(data_x)
            reduced_x6 = pca.transform(data_x)
        elif val == str(9):
            df = buckets_df[val]
            #dff = []
            #for file in df:
            #    if 'center' in file[1]:
             #       dff.append(file)
            data_x, data_y = rep_layer_ouptut(df, rep_layer)
            pca.fit(data_x)
            reduced_x9 = pca.transform(data_x)
    plt.scatter(reduced_x6.transpose()[0],reduced_x6.transpose()[1])
    plt.scatter(reduced_x9.transpose()[0],reduced_x9.transpose()[1])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_x6.transpose()[0],reduced_x6.transpose()[1], reduced_x6.transpose()[2])
    ax.scatter(reduced_x9.transpose()[0],reduced_x9.transpose()[1],reduced_x9.transpose()[2])
    plt.legend(['x6', 'x9'])
