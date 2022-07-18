import numpy as np
from tqdm import tqdm
from utils import zscore_masked_items, get_rmse_score
from fancyimpute import SoftImpute, BiScaler


class SOFTIMPUTE():

    def __init__(self, A, max_rank=3, max_iters=10, n_power_iterations=100):
        self.A = A
        self.W = (self.A > 0).astype(int)
        self.num_users, self.num_items = self.A.shape
        self.max_rank = max_rank
        self.max_iters = max_iters
        self.n_power_iterations = n_power_iterations
        self.solver = None
        self.scaler = None

        self.norm_A, self.mean_A, self.stddev_A = zscore_masked_items(self.A, self.W)
        self.A_t = np.zeros((self.num_users, self.num_items))

    def train(self):
        self.solver = SoftImpute(max_rank=self.max_rank, max_iters=self.max_iters,
                                 n_power_iterations=self.n_power_iterations
                                 )
        self.scaler = BiScaler(min_value=1, max_value=5)

    def reconstruct_matrix(self):
        """
        Compute the full matrix using the solver
        """
        data_normalized = self.scaler.fit_transform(self.A)
        prediction_norm = self.solver.fit_transform(data_normalized)
        prediction_matrix = self.scaler.inverse_transform(prediction_norm)
        return prediction_matrix
