from model import DataManager

mat_structure = DataManager.load_mat_file("data/reuters.mat")
columns_name, attributes = DataManager.extract_data(mat_structure, 'WORDS')

