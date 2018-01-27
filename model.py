import scipy.io
import numpy as np


class DataManager:
    def __init__(self):
        pass

    @staticmethod
    def load_mat_file(file_path):
        mat_structure = scipy.io.loadmat(file_path)
        return mat_structure

    @staticmethod
    def extract_data(mat_structure, attribute_name):
        words_array = scipy.sparse.csc_matrix(mat_structure[attribute_name]).toarray()
        words_array = np.asmatrix(words_array)
        words_column_names = list()
        for list_item in mat_structure[attribute_name + '_COLUMN_NAMES'][0]:
            words_column_names.append(str(list_item[0]))
        if words_array.shape[1] != len(words_column_names):
            raise ValueError('Length of columns name and attributes must be the same!')
        return words_column_names, words_array
