from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

NUMBER_OF_USERS = 10000
NUMBER_OF_MOVIES = 1000
RAND_SEED = 42

def extract_for_ensemble(matrix, fname, fold, train):
    full_path = "./data/ensemble/final/"+fname
    if train:
        full_path = "./data/ensemble/train/"+str(fold)+"/"+fname
    np.save(full_path, matrix)

def split_dataset(data, train_size=0.9, random_state=None):
    # Split the dataset into train and test
    train_pd, test_pd = train_test_split(data, train_size=train_size, random_state=random_state)
    return train_pd, test_pd

def get_rmse_score(predictions, target_values):
    # test our predictions with the true values
    test_users, test_items = target_values.nonzero()
    y_pred = []
    y_true = []
    for user, item in zip(test_users, test_items):
        y_pred.append(predictions[user][item])
        y_true.append(target_values[user][item])
    return mean_squared_error(y_pred, y_true, squared=False)

def create_matrix_from_raw(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in \
        np.split(data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    matrix = np.full((NUMBER_OF_USERS, NUMBER_OF_MOVIES), 0)
    for user , movie , pred in zip(users , movies ,predictions):
        matrix[user][movie] = pred
    return matrix

def import_data_to_matrix_split(train_size=0.9, random_state=None) -> np.ndarray:
    # Extract data to row-column format
    data_pd = pd.read_csv("./data/data_train.csv")
    train_pd, test_pd = split_dataset(data_pd, train_size=train_size, random_state=random_state)
    train_matrix = create_matrix_from_raw(train_pd)
    test_matrix = create_matrix_from_raw(test_pd)
    return train_matrix, test_matrix

def import_data_to_matrix() -> np.ndarray:
    # Extract data to row-column format
    data_pd = pd.read_csv("./data/data_train.csv")
    matrix = create_matrix_from_raw(data_pd)
    return matrix

def extract_submission(matrix: np.ndarray, file: str = "sumbission"):
    # Extract sumbission users and movies
    sample_pd = pd.read_csv("./data/sampleSubmission.csv")
    test_users, test_movies = \
        [np.squeeze(arr) for arr in \
        np.split(sample_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1)]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    with open("./data/submissions/" + file + "_" + timestamp + ".csv", "w") as f:
        f.write("Id,Prediction\n")
        for (user, movie) in zip(test_users, test_movies):
            f.write("r{}_c{},{}\n".format(user + 1, movie + 1, matrix[user,movie]))

def zscore_masked_items(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm_matrix = np.copy(matrix).astype(float)
    maskT = mask.T
    means = []
    stddevs = []
    for j, matrixj in enumerate(matrix.T):
        mean = np.mean(matrixj[maskT[j]!=0])
        std = np.std(matrixj[maskT[j]!=0])
        means.insert(j, mean)
        stddevs.insert(j, std)
        norm_matrix[:,j][np.where(mask[:,j]!=0)] -= mean
        norm_matrix[:,j][np.where(mask[:,j]!=0)] /= std
    return norm_matrix, means, stddevs

def zscore_masked_users(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    norm_matrix = np.copy(matrix).astype(float)
    for i, matrixi in enumerate(matrix):
        mean = np.mean(matrixi[mask[i]!=0])
        std = np.std(matrixi[mask[i]!=0])
        norm_matrix[i][np.where(mask[i]!=0)] -= mean
        norm_matrix[i][np.where(mask[i]!=0)] /= std
    return norm_matrix

def zscore_items(matrix: np.ndarray) -> np.ndarray:
    norm_matrix = matrix - np.mean(matrix, axis = 0)
    norm_matrix /= np.std(matrix, axis = 0)
    return norm_matrix

def zscore_users(matrix: np.ndarray) -> np.ndarray:
    norm_matrix = matrix.T - np.mean(matrix, axis = 1)
    norm_matrix /= np.std(matrix, axis = 1)
    return norm_matrix.T
