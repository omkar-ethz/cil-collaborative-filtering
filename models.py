import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import NUMBER_OF_MOVIES, NUMBER_OF_USERS, import_data_to_matrix_split, extract_submission, import_data_to_matrix
from utils import get_rmse_score, zscore_masked_items

class IRSVD():

    def __init__(self, A, biases="mean", features=324, eta=0.01, lambda1=0.02, lambda2=0.05, epochs=15):
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
        self.lambda2 = lambda2
        self.epochs = epochs
        
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale=1./self.features, size=(self.num_users, self.features))
        self.V = np.random.normal(scale=1./self.features, size=(self.num_items, self.features))
        
        # Initialize the biases
        self.global_mean = np.sum(self.W * self.A)/np.sum(self.W)
        if biases == "zero":
            self.Bu = np.zeros(self.num_users)
            self.Bi = np.zeros(self.num_items)
        else:
            Mu = np.array([np.sum(Wu * Au)/np.sum(Wu) for Au, Wu in zip(self.A, self.W)])
            Mi = np.array([np.sum(Wi * Ai)/np.sum(Wi) for Ai, Wi in zip(self.A.T, self.W.T)])

            self.Bu = Mu - np.mean(Mu)
            self.Bi = Mi - np.mean(Mi)

        self.Bu = np.reshape(self.Bu, (self.Bu.shape[0],1))
        self.Bi = np.reshape(self.Bi, (self.Bi.shape[0],1))

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
            prediction = self.global_mean + self.Bu[user] + self.Bi[item] + np.dot(self.U[user, :], self.V[item, :].T)
            error = (rating - prediction)

            # Update biases
            self.Bu[user] += self.eta * (error - self.lambda2 * self.Bu[user])
            self.Bi[item] += self.eta * (error - self.lambda2 * self.Bi[item])

            # Update user and item feature matrices
            temp_U = np.copy(self.U[user, :])
            self.U[user, :] += self.eta * (error * self.V[item, :] - self.lambda1 * self.U[user,:])
            self.V[item, :] += self.eta * (error * temp_U - self.lambda1 * self.V[item,:])

    def reconstruct_matrix(self):
        """
        Compute the reconstructed matrix using biases, U and V
        """
        biases = self.global_mean + np.array([self.Bu.T[0]]*self.num_items).T + np.array([self.Bi.T[0]]*self.num_users)
        return biases + np.dot(self.U, self.V.T)


class GBias():
    
    def __init__(self, A, lambda1=0.001, epochs=5):
        self.A = A
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.lambda1 = lambda1
        self.epochs = epochs
        
        # Initialize the biases
        self.global_mean = np.sum(self.W * self.A)/np.sum(self.W)
        Mu = np.array([np.sum(Wu * Au)/np.sum(Wu) for Au, Wu in zip(self.A, self.W)])
        Mi = np.array([np.sum(Wi * Ai)/np.sum(Wi) for Ai, Wi in zip(self.A.T, self.W.T)])

        self.Bu = Mu - np.mean(Mu)
        self.Bi = Mi - np.mean(Mi)

        self.Bu = np.reshape(self.Bu, (self.Bu.shape[0],1))
        self.Bi = np.reshape(self.Bi, (self.Bi.shape[0],1))
    
    def _loss(self):
        return ((1/2) * np.sum((self.W * (self.A - self.global_mean - np.dot(self.Bu, np.ones((1, self.num_items))) - np.dot(self.Bi, np.ones((1, self.num_users))).T) ** 2))
                + (self.lambda1/2) * (np.sum(self.Bu ** 2) + np.sum(self.Bi ** 2)))
    
    def train(self, test_matrix=None):
        error_progress = {
            "train_rmse": [],
            "test_rmse": [],
        }
        for epoch in tqdm(range(self.epochs)):
            self._als()
            rec_A = self.reconstruct_matrix()
            train_rmse = get_rmse_score(rec_A, self.A)
            error_progress["train_rmse"].append(train_rmse)
            if test_matrix is not None:
                test_rmse = get_rmse_score(rec_A, test_matrix)
                error_progress["test_rmse"].append(test_rmse)
            # print(error_progress)
        return error_progress
    
    def _als(self):
        self.Bi = np.linalg.solve(self.lambda1 + np.diag(np.dot(self.W.T, np.ones((self.num_users,1))).T[0]),
                                    np.dot(
                                        self.W.T * (self.A.T - self.global_mean - np.dot(np.ones((self.num_items, 1)), self.Bu.T)),
                                        np.ones((self.num_users, 1))
                                    )
                                )
        # print("Loss l(Bu,Bi) after solving for Bi:", self._loss())

        self.Bu = np.linalg.solve(self.lambda1 + np.diag(np.dot(self.W, np.ones((self.num_items,1))).T[0]),
                                np.dot(
                                    self.W * (self.A - self.global_mean - np.dot(np.ones((self.num_users, 1)), self.Bi.T)),
                                    np.ones((self.num_items, 1))
                                )
                            )
        # print("Loss l(Bu,Bi) after solving for Bu:", self._loss())
    
    def reconstruct_matrix(self):
        """
        Compute the full matrix using biases
        """
        biases = self.global_mean + np.array([self.Bu.T[0]]*self.num_items).T + np.array([self.Bi.T[0]]*self.num_users)
        return biases


class Baseline():
    
    def __init__(self, A, K=3, lambda1=0.1, epochs=3):
        self.A = A
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.K = K
        self.lambda1 = lambda1
        self.epochs = epochs
        
        self.norm_A, self.mean_A, self.stddev_A = zscore_masked_items(self.A, self.W)

        # SVD decomposition init U and V
        U, s, Vt = np.linalg.svd(self.norm_A, full_matrices=False)
        # Using the top k eigenvalues
        S = np.zeros((self.num_items , self.num_items))
        S[:self.K, :self.K] = np.diag(s[:self.K])
        # Initialize ALS with SVD result
        # Only first k columns the rest are all set to zero
        self.U = U.dot(S)[:,:self.K]
        self.V = S.dot(Vt)[:self.K]
    
    def _loss(self):
        return ((1/2) * np.sum((self.W * (self.A - np.dot(self.U, self.V)) ** 2))
                + (self.lambda1/2) * (np.sum(self.U ** 2) + np.sum(self.V ** 2)))
    
    def train(self, test_matrix=None):
        error_progress = {
            "train_rmse": [],
            "test_rmse": [],
        }
        for epoch in tqdm(range(self.epochs)):
            self._als()
            rec_A = self.reconstruct_matrix()
            train_rmse = get_rmse_score(rec_A, self.A)
            error_progress["train_rmse"].append(train_rmse)
            if test_matrix is not None:
                test_rmse = get_rmse_score(rec_A, test_matrix)
                error_progress["test_rmse"].append(test_rmse)
            # print(error_progress)
        return error_progress
    
    def _als(self):
        for j, Wj  in enumerate(self.W.T):
            self.V[:,j] = np.linalg.solve(np.dot(self.U.T, np.dot(np.diag(Wj), self.U)) + self.lambda1 * np.eye(self.K),
                                    np.dot(self.U.T, np.dot(np.diag(Wj), self.norm_A[:, j])))
        # print("Loss l(U,V) after solving for V:", self._loss())

        for i, Wi  in enumerate(self.W):
            self.U[i] = np.linalg.solve(np.dot(self.V, np.dot(np.diag(Wi), self.V.T)) + self.lambda1 * np.eye(self.K),
                                np.dot(self.V, np.dot(np.diag(Wi), self.norm_A[i].T))).T
        # print("Loss l(U,V) after solving for U:", self._loss())
    
    def reconstruct_matrix(self):
        """
        Compute the full matrix using U and V from als and undo normilization.
        """
        rec_A = np.dot(self.U, self.V)
        #undo normalization
        for j in range(self.num_items):
            rec_A[:,j] *= self.stddev_A[j]
            rec_A[:,j] += self.mean_A[j]
        return rec_A


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


class SVP():
    
    def __init__(self, A, eta=5, K=3, epochs=10):
        self.A = A
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.eta = eta
        self.K = K
        self.epochs = epochs
        
        self.norm_A, self.mean_A, self.stddev_A = zscore_masked_items(self.A, self.W)
        self.A_t = np.zeros((self.num_users , self.num_items))

    def train(self, test_matrix=None):
        error_progress = {
            "train_rmse": [],
            "test_rmse": [],
        }
        for epoch in tqdm(range(self.epochs)):
            self._pgd()
            self.eta = self.eta/(epoch+1)**(1/2)
            rec_A = self.reconstruct_matrix()
            train_rmse = get_rmse_score(rec_A, self.A)
            error_progress["train_rmse"].append(train_rmse)
            if test_matrix is not None:
                test_rmse = get_rmse_score(rec_A, test_matrix)
                error_progress["test_rmse"].append(test_rmse)
            # print(error_progress)
        return error_progress
    
    def _pgd(self):
        residual = self.norm_A - self.A_t
        temp = self.A_t + self.eta * self.W * residual

        # SVD
        U, s, Vt = np.linalg.svd(temp, full_matrices=False)
        S = np.zeros((self.num_items , self.num_items))
        S[:self.K, :self.K] = np.diag(s[:self.K])
        # Reconstruction - project temp to space of rank k matrices using SVD computed above
        self.A_t = np.dot(np.dot(U, S), Vt)

    def reconstruct_matrix(self):
        """
        Compute the full matrix using A_t and undo normalization
        """
        rec_A = np.copy(self.A_t)
        #undo normalization
        for j in range(self.num_items):
            rec_A[:,j] *= self.stddev_A[j]
            rec_A[:,j] += self.mean_A[j]
        return rec_A


class SVT():
    
    def __init__(self, A, eta=1.2, tau=2000, epochs=28):
        self.A = A
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.eta = eta
        self.tau = tau
        self.epochs = epochs
        
        self.norm_A, self.mean_A, self.stddev_A = zscore_masked_items(self.A, self.W)
        self.A_t = np.zeros((self.num_users , self.num_items))
    
    def train(self, test_matrix=None):
        error_progress = {
            "train_rmse": [],
            "test_rmse": [],
        }
        for epoch in tqdm(range(self.epochs)):
            self._shrinkgd()
            # self.eta = self.eta/(epoch+1)**(1/2)
            rec_A = self.reconstruct_matrix()
            train_rmse = get_rmse_score(rec_A, self.A)
            error_progress["train_rmse"].append(train_rmse)
            if test_matrix is not None:
                test_rmse = get_rmse_score(rec_A, test_matrix)
                error_progress["test_rmse"].append(test_rmse)
            # print(error_progress)
        return error_progress

    def _shrinkgd(self):
        shrinked_A_t = self._shrink()
        self.A_t = self.A_t + self.eta * self.W * (self.norm_A - shrinked_A_t)

    def _shrink(self):
        U, s, Vt = np.linalg.svd(self.A_t, full_matrices=False)
        # print(s[:10])
        s = s - self.tau
        # print("s-tau", s[:10])
        s[s < 0] = 0 #clip singular values
        # print("s clipped", s[:10])
        return np.dot(U * s, Vt)

    def reconstruct_matrix(self):
        """
        Compute the full matrix using last A_t and perform the shrinkage op and
        undo normalization.
        """
        shrinked_A_t = self._shrink()
        rec_A = shrinked_A_t
        #undo normalization
        for j in range(self.num_items):
            rec_A[:,j] *= self.stddev_A[j]
            rec_A[:,j] += self.mean_A[j]
        return rec_A

import mxnet as mx
# Deep Matrix Factorization
class DMF():
    def __init__(self, A, batch_size=1024, dim_user=15, dim_movie=15, epochs=10):
        self.A = A
        self.train_users, self.train_items = self.A.nonzero()
        self.train_ratings = A[self.train_users, self.train_items]
        self.num_users, self.num_items = self.A.shape
        self.batch_size = batch_size
        self.dim_user = dim_user
        self.dim_movie = dim_movie
        self.epochs = epochs


    def train(self, test_matrix=None):
        X_train = mx.io.NDArrayIter({'user': self.train_users, 'movie': self.train_items}, 
                            label=self.train_ratings, batch_size=self.batch_size)
        
        if test_matrix is not None:
            test_users, test_items = test_matrix.nonzero()
            test_ratings = test_matrix[test_users, test_items]
            X_eval = mx.io.NDArrayIter({'user': test_users, 'movie': test_items}, 
                            label=test_ratings, batch_size=self.batch_size)
        else:
            X_eval = None
        user = mx.symbol.Variable("user")
        user = mx.symbol.Embedding(data=user, input_dim=self.num_users, output_dim=self.dim_user)

        movie = mx.symbol.Variable("movie")
        movie = mx.symbol.Embedding(data=movie, input_dim=self.num_items, output_dim=self.dim_movie)

        y_true = mx.symbol.Variable("softmax_label")

        nn = mx.symbol.concat(user, movie)
        nn = mx.symbol.flatten(nn)
        nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)
        nn = mx.symbol.BatchNorm(data=nn) # First batch norm layer here, before the activaton!
        nn = mx.symbol.Activation(data=nn, act_type='relu') 
        nn = mx.symbol.FullyConnected(data=nn, num_hidden=64)
        nn = mx.symbol.BatchNorm(data=nn) # Second batch norm layer here, before the activation!
        nn = mx.symbol.Activation(data=nn, act_type='relu')
        nn = mx.symbol.FullyConnected(data=nn, num_hidden=1)

        y_pred = mx.symbol.LinearRegressionOutput(data=nn, label=y_true)

        self.model = mx.module.Module(context=mx.cpu(), data_names=('user', 'movie'), symbol=y_pred)
        self.model.fit(X_train, num_epoch=self.epochs, optimizer='adam', optimizer_params=(('learning_rate', 0.001),),
                eval_metric='rmse', eval_data=X_eval, batch_end_callback=mx.callback.Speedometer(self.batch_size, 256))

    
    def reconstruct_matrix(self):
        """
        Compute the full matrix using A_t and undo normalization
        """
        indices = np.indices((NUMBER_OF_USERS,NUMBER_OF_MOVIES))
        total = NUMBER_OF_USERS * NUMBER_OF_MOVIES
        X_test = mx.io.NDArrayIter({'user': indices[0].reshape(total), 
            'movie': indices[1].reshape(total)}, batch_size=self.batch_size)
        test_preds = self.model.predict(X_test)
        rec_A = test_preds.asnumpy().reshape((NUMBER_OF_USERS, NUMBER_OF_MOVIES))
        return rec_A