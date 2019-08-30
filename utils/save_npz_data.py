import numpy as np
from keras.applications.vgg16 import VGG16

from core.dataset import DataSet
from core.knn import reset_weights
from utils.data_processing import rep_layer_ouptut

vgg16_rep_layer = VGG16(weights=None, include_top=False, pooling='max', input_shape=(101, 101, 3))
reset_weights(vgg16_rep_layer)

data_001 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                   train_subsets=[],
                   test_subsets=['/001/'],
                   location='local')

data_002 = DataSet(root_dir='/Users/jesusnavarro/Desktop/DataSet/',
                   train_subsets=[],
                   test_subsets=['/002/'],
                   location='local')

print(vgg16_rep_layer.summary())

d1_df = data_001.test_set[2]
d2_df = data_002.test_set[2]

x1, y1 = rep_layer_ouptut(d1_df, vgg16_rep_layer)
x2, y2 = rep_layer_ouptut(d2_df, vgg16_rep_layer)

path = '/Users/jesusnavarro/Desktop/vgg16_tests/npz_data/vgg16_output_top_off_pooling_resized.npz'
np.savez(path, x1=x1, y1=y1, x2=x2, y2=y2)
print(vgg16_rep_layer.summary())
