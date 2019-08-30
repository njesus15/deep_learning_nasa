import os
import numpy as np
from imageio import imread
from keras.preprocessing import image
from tensorflow.python.lib.io import file_io
import pickle
from itertools import product

class DataSet():

    """ A class for reading the trail forest dataset

    Attributes:
        location (str): Training location either local or google cloud
        root_directory (str): local root directory of trail forest dataset
        train_set (list, len = 3 ): Dataset containing data in an array (optional)
                                            and images paths with class labels
        test_set (list, len = 3): Similar to train set but with test data
        data_struct (dictionary): Depicts training and test dataset names for forest trail data
    """

    def __init__(self, root_dir, train_subsets, test_subsets, type='list', loc='local'):
        """ The constructor for the class DataSet

        :param root_dir: local directory of dataset
        :param train_subsets: list of strings containing which dataset to train
        :param test_subsets: list of strings containing which dataset to test
        :param type: str to choose opt of train_set and test_set att. to contain numpy array of images
        :param loc: location to train the model: 'gs': google cloud, "local": local machine

        TODO: Replace variable 'type' with boolean instance variable i.e. array_bool
        """
        self.array_bool = not type == 'list'
        self.location = loc
        self.root_directory = root_dir
        self.train_set = self.array_data(train_subsets, self.array_bool)
        self.test_set = self.array_data(test_subsets, self.array_bool)
        self.data_struct = {'train': train_subsets, 'test': test_subsets}


    def array_data(self, dataframe, array_bool):
        """

        :param dataframe: List of tuples containing image paths and class names
        :param array_bool: True/False for outputting data in array (large memory consumption)
        :return: List containing image array, label encoding, and paths
        """
        data = self.get_trail_dataframe(dataframe, array_bool)
        img_array = [file[0] for file in data]
        y_array = [file[1] for file in data]
        paths = [file[2] for file in data]

        return [np.array(img_array), np.array(y_array), paths]

    def get_data_frame(self, path_names, target_subset, array_bool, use_small=True):
        """ Helper function to extract target data

        :param path_names: Hard coded path names to fit file structure of downloaded dataset
        :param target_subset: Subset of data names (i.e. '/001/') to extract
        :param array_bool: Boolean for array data of images
        :param use_small: Boolean for compressed images
        :return: data in list format containing [image array, vector encoding class, [path, label]] per item

        TODO: Eliminate param array_bool and use_small
        """
        path_names = path_names
        data_frame = []

        # If gs read from google storage bucket
        paths = []
        if self.location == 'gs':
            # Different image sizes: Dataset (101,101,3), DataSet2 (224,224,3) used for vgg16
            if '/DataSet/' in self.root_directory:
                file_stream = file_io.FileIO("gs://data-daisy/full_gs_paths_subset3.pickle", mode='rb')
            else:
                file_stream = file_io.FileIO("gs://data-daisy/full_gs_paths_large_size.pickle", mode='rb')
            paths = pickle.load(file_stream)

            data_frame = self.read_from_full_paths(paths, target_subset)

        else: # Read from local machine
            for class_label in path_names.keys():
                for dir_path in path_names[class_label]:
                    paths.append([self.collect_image_paths(dir_path), class_label])  # hard coded to fit file structure of dataset
            corrected_path_df = []
            for path in paths: # Different structure (i.e., [([filenames], class)....]
                class_ = path[1]
                for file_name in path[0]:
                    corrected_path_df.append([file_name, class_])
            data_frame = self.read_from_full_paths(corrected_path_df, target_subset)
        return data_frame

    def collect_image_paths(self, path, use_small=True):
        """ Returns the full image paths given the root directory of the dataset.
            Note that this was hard coded to fit the unmodified file strcuture of the
            dataset. [Source: http://people.idsia.ch/~guzzi/DataSet.html]

        :param path: sub root directories define in get_trail_data_frame
        :param use_small: Boolean to use compressed images
        TODO: Remve use_small boolean
        """
        paths = []
        root_path = self.root_directory
        full_path = os.path.join(root_path, path)  # joins the
        for sub_item in os.listdir(full_path):
            if os.path.isdir(os.path.join(full_path, sub_item)):  # checks that subitem is a valid directory
                for file_name in os.listdir(
                        os.path.join(full_path, sub_item)):  # iterates through each file nice of each subitem
                    # Seems like that joined path is appended regardless of logic
                    if use_small:
                        if file_name.endswith(".jpg") and 'tiny' in file_name:
                            paths.append(
                                os.path.join(self.root_directory, path, sub_item, file_name))

        return paths

    def save_small_images(self, image_size=(101, 101)):
        """ Method to compress images from trail forest dataset

        :param image_size: Compressed image size to save image. Default is set to 101, 101 to fit ConvNet
        :return: None

        TODO: Fix bug
        """
        i = 0.0
        for item in self.full_paths:
            i += 1
            image_path, image_name = os.path.split(item[0])
            thumb_full_path = os.path.join(image_path, 'tiny_' + image_name)
            if not os.path.exists(thumb_full_path):
                small_image = image.load_img(item[0], target_size=image_size)
                small_image.save(thumb_full_path)
            item[0] = thumb_full_path

    def get_trail_dataframe(self, target_subset, use_small=True):
        """

        :param target_subset (list of str) : Subset of images to extract
        :param use_small (boolean) : Boolean to use compressed images
        :return: Dataframe containing data
        """

        # Hard coded sub-directories identical to file structure of dataset

        path_names = {"right": ["000/videos/rc", "001/videos/rc", "002/videos/rc", "003/videos/rc", "004/videos/rc",
                                "005/videos/rc", "006/videos/rc",
                                "007/videos/rc", "008/videos/rc", "009/videos/rc", "010/videos/rc", "011/videos/rc"],
                      "center": ["000/videos/sc", "001/videos/sc", "002/videos/sc", "003/videos/sc", "004/videos/sc",
                                 "005/videos/sc", "006/videos/sc",
                                 "007/videos/sc", "008/videos/sc", "009/videos/sc", "010/videos/sc", "011/videos/sc"],
                      "left": ["000/videos/lc", "001/videos/lc", "002/videos/lc", "003/videos/lc", "004/videos/lc",
                               "005/videos/lc", "006/videos/lc", "007/videos/lc",
                               "008/videos/lc", "009/videos/lc", "010/videos/lc", "011/videos/lc"]}
        data_frame = self.get_data_frame(path_names, target_subset, use_small)
        return data_frame

    def read_from_full_paths(self, paths, target_subset, class_label=None):
        data_frame=[]

        # one hot encoding key
        key = {'right': np.array([1, 0, 0]),
               'center': np.array([0, 1, 0]),
               'left': np.array([0, 0, 1])}

        for path in paths:
            class_ = path[1]
            file_name = path[0]
            img_array = []

            # check if type list
            if not isinstance(target_subset, list):
                target_subset = [target_subset]

            if any(target_path in file_name for target_path in target_subset) and 'tiny' in file_name:
                if self.array_bool:
                    with file_io.FileIO(file_name, 'rb') as f:
                        img_array = f.read()
                        img_array = imread(img_array) * 1.0 / 255.0
                y_label = key[class_]
                data_frame.append([img_array, y_label, [file_name, class_]])

        return data_frame