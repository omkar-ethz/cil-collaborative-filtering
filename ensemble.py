import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from utils import create_matrix_from_raw, user_movies_pred, extract_submission
from utils import RAND_SEED

class Ensemble():
    
    def __init__(self, fold=None):
        self.num_folds = 10
        self.fold = None
        if fold is not None:
            if fold not in range(1,11):
                self.fold = 1
            else:
                self.fold = fold
            self.num_folds = 1
        self.data_pd = pd.read_csv("./data/data_train.csv")
        self.kf = KFold(n_splits=10, shuffle=True, random_state=RAND_SEED)
        self.weights = None
        self.avg_weights = None
    
    def train(self):
        for idx, (train_set, val_set) in enumerate(self.kf.split(self.data_pd)):
            if self.fold is not None and self.fold != idx+1: continue
            train_data = self.data_pd.iloc[train_set]
            val_data = self.data_pd.iloc[val_set]
            
            train_matrix = create_matrix_from_raw(train_data)
            val_users, val_movies, val_pred = user_movies_pred(val_data)
                        
            X_ensemble = np.ones((1, val_pred.shape[0]))
            
            path = "./data/ensemble/train/"+str(idx+1)+"/"
            for file in os.listdir(path):
                if file.endswith(".npy"):
                    rec_A = np.load(os.path.join(path, file))
                    pred = rec_A[val_users, val_movies]
                    X_ensemble = np.vstack([X_ensemble, pred])
            X_ensemble = X_ensemble.T
            
            b, _, _, _ = np.linalg.lstsq(X_ensemble, val_pred, rcond=None)
            if self.weights is None:
                self.weights = b
            else:
                self.weights = np.vstack([self.weights, b])
        
        if self.num_folds == 1:
            self.avg_weights = np.copy(self.weights)
        else:
            self.avg_weights = self.weights.sum(axis=0)/self.num_folds
    
    def reconstruct_matrix(self, on_fold=False):
        """
        Reconstruct matrix based on the ensemble weights, on full trained models
        """
        rec_A = None
        path = "./data/ensemble/final/"
        if on_fold and self.fold is not None:
            path = "./data/ensemble/train/"+str(self.fold)+"/"
        idx = 1
        for file in os.listdir(path):
            if file.endswith(".npy"):
                if rec_A is None:
                    rec_A = np.load(os.path.join(path, file)) * self.avg_weights[idx]
                else:
                    rec_A += np.load(os.path.join(path, file)) * self.avg_weights[idx]
                idx += 1
        rec_A += np.ones((rec_A.shape[0], rec_A.shape[1])) * self.avg_weights[0]
        return rec_A
