import numpy as np
from tqdm import tqdm

def rmse(A, A_t, W):
    return np.sum(W*((A-A_t)**2)) / np.sum(W)

def svp(norm_A, W, k):
    iter = 1
    A_t = np.zeros((10000 , 1000))
    eta = 5.0
    for epoch in tqdm(range(5)):
        residual = norm_A - A_t
        temp = A_t + eta * W * residual

        # SVD
        U, s, Vt = np.linalg.svd(temp, full_matrices=False)
        s[k+1:] = 0
        # Reconstruction - project temp to space of rank k matrices using SVD computed above
        A_t = np.dot(U * s, Vt)

        iter+=1
        eta = eta / (iter)**(1/2)
        print("k = ", k, ", RMSE = ", rmse(norm_A, A_t, W))
    return A_t