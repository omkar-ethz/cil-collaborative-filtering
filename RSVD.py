import numpy as np
from tqdm import tqdm
from utils import get_rmse_score

class RSVD():

    def __init__(self, A, features=324, eta=0.01, lambda1=0.02, epochs=15):
        """
        Perform matrix decomposition to predict empty
        entries in a matrix.
        """
        self.A = A
        train_users, train_items = self.A.nonzero()
        self.train_entries = [(user, item, self.A[user][item]) 
                              for user, item in zip(train_users, train_items)]
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.features = features
        self.eta = eta
        self.lambda1 = lambda1
        self.epochs = epochs
        
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale=1./self.features, size=(self.num_users, self.features))
        self.V = np.random.normal(scale=1./self.features, size=(self.num_items, self.features))

    def train(self, test_matrix=None):
        # Perform stochastic gradient descent for number of epochs
        error_progress = {
            "train_rmse": [],
            "test_rmse": [],
        }
        for epoch in tqdm(range(self.epochs)):
            # shuffling will help during training
            np.random.shuffle(self.train_entries)
            # print("Entering sgd")
            self._sgd()
            # print("Finishing sgd")
            rec_A = self.reconstruct_matrix()
            train_rmse = get_rmse_score(rec_A, self.A)
            error_progress["train_rmse"].append(train_rmse)
            if test_matrix is not None:
                test_rmse = get_rmse_score(rec_A, test_matrix)
                error_progress["test_rmse"].append(test_rmse)
            # print(error_progress)
        return error_progress

    def _sgd(self):
        """
        Perform stochastic gradient descent
        """
        for user, item, rating in self.train_entries:
            # Compute prediction and error
            prediction = np.dot(self.U[user, :], self.V[item, :].T)
            error = (rating - prediction)

            # Update user and item feature matrices
            temp_U = np.copy(self.U[user, :])
            self.U[user, :] += self.eta * (error * self.V[item, :] - self.lambda1 * self.U[user,:])
            self.V[item, :] += self.eta * (error * temp_U - self.lambda1 * self.V[item,:])

    def reconstruct_matrix(self):
        """
        Compute the reconstructed matrix using U and V
        """
        return np.dot(self.U, self.V.T)