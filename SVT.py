import numpy as np
from tqdm import tqdm
from utils import zscore_masked_items, get_rmse_score

class SVT():
    
    def __init__(self, A, eta=1.2, tau=2000, epochs=25):
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