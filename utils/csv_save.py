from keras import Model
from keras.engine.saving import load_model
import pickle
from trainer.DataSet import DataSet
import numpy as np
import csv

def save_cvs():
    model_exc_001 = load_model('/Users/jesusnavarro/Desktop/trail_project/results/trained_models/deepnn_subset_001.h5.h5')


    model_exc_001_rep =  Model(inputs=model_exc_001.input,
                                outputs=model_exc_001.get_layer(index=7).output)

    dataset_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                       train_subsets=[],
                       test_subsets=['/001/'])

    dataset_002 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                       train_subsets=[],
                       test_subsets=['/002/'])

    key = {str(np.array([1, 0, 0])):'right',
               str(np.array([0, 1, 0])): 'center',
               str(np.array([0, 0, 1])): 'left'}

    csv_001_key = {'left': 0, 'center': 1, 'right': 2}
    csv_002_key = {'left': 3, 'center': 4, 'right': 5}

    x_001, y_001, paths_001 = dataset_001.test_set
    x_002, y_002, paths_002 = dataset_002.test_set

    int_001_output = model_exc_001_rep.predict(x_001)
    int_001_output = int_001_output.reshape(x_001.shape[0], - 1)

    int_002_output = model_exc_001_rep.predict(x_002)
    int_002_output = int_002_output.reshape(x_002.shape[0], - 1)


    csvData = []

    for int_out, paths, y in zip(int_001_output, paths_001, y_001):
        dir_class = key[str(y)]
        label = csv_001_key[dir_class]
        tmp_str = ''
        for val in int_out:
            tmp_str += str(val) + ','
        tmp_str = paths + '\t' + str(label) + '\t' + '[' + tmp_str[:-1] + ']' + '\n'
        csvData.append(tmp_str)

    for int_out, paths, y in zip(int_002_output, paths_002, y_002):
        dir_class = key[str(y)]
        label = csv_002_key[dir_class]
        tmp_str = ''
        for val in int_out:
            tmp_str += str(val) + ','
        tmp_str = paths + '\t' + str(label) + '\t' + '[' + tmp_str[:-1] + ']' + '\n'
        csvData.append(tmp_str)


    f = open('/Users/jesusnavarro/Desktop/trail_project/Pickledata/visualize_001_002_nn_exc_001.txt', 'w+')

    for line in csvData:
        print(line)
        f.write(line)
    f.close()
    return None

model_exc_001 = load_model('/Users/jesusnavarro/Desktop/trail_project/results/trained_models/deepnn_subset_full_exc_001.h5')


dataset_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                      train_subsets=[],
                      test_subsets=['/001/'])

dataset_002 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                      train_subsets=[],
                      test_subsets=['/002/'])

key = {str(np.array([1, 0, 0])): 'right',
       str(np.array([0, 1, 0])): 'center',
       str(np.array([0, 0, 1])): 'left'}

csv_001_key = {'left': 0, 'center': 1, 'right': 2}
csv_002_key = {'left': 3, 'center': 4, 'right': 5}

x_001, y_001, paths_001 = dataset_001.test_set
x_002, y_002, paths_002 = dataset_002.test_set

_, acc_001 = model_exc_001.evaluate(x_001, y_001)
_, acc_002 = model_exc_001.evaluate(x_002, y_002)

print(acc_001, acc_002)
