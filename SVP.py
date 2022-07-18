import numpy as np
from tqdm import tqdm
from utils import zscore_masked_items, get_rmse_score

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