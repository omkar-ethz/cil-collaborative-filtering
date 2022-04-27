from datetime import datetime

import pandas as pd
import numpy as np

NUMBER_OF_USERS = 10000
NUMBER_OF_MOVIES = 1000

def import_data_to_matrix() -> np.ndarray:
    # Extract data to row-column format
    data_pd = pd.read_csv("./data/data_train.csv")
    users, movies = \
        [np.squeeze(arr) for arr in \
        np.split(data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    data = np.full((NUMBER_OF_USERS, NUMBER_OF_MOVIES), 0)
    for user , movie , pred in zip(users , movies ,predictions):
        data[user][movie] = pred
    return data

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
    for j, matrixj in enumerate(matrix.T):
        mean = np.mean(matrixj[maskT[j]!=0])
        std = np.std(matrixj[maskT[j]!=0])
        norm_matrix[:,j][np.where(mask[:,j]!=0)] -= mean
        norm_matrix[:,j][np.where(mask[:,j]!=0)] /= std
    return norm_matrix

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
