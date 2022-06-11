#!/usr/bin/env python
# coding: utf-8

# # Baseline Method
# - Impute missing data with 0
# - Normalize item by item (Only observed entries)
# - SVD with $k = 3$
# - Use result of SVD to initialize ALS (UV)
# - Run ALS with $k = 3$ and $λ = 0.1$ for 20 iterations

# ## Packages

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import import_data_to_matrix, extract_submission
from utils import NUMBER_OF_MOVIES, NUMBER_OF_USERS
from utils import zscore_masked_items


# ## Basic Settings
# - $k = 3$, $λ = 0.1$, 20 ALS (UV) iterations 

# In[ ]:


k = 3 #number of top eigenvalues of SVD, choose by cross validation
lambda_ = 0.1
iterations = 20


# ## Data Preprocessings
# - Extract data to row-column format
# - Impute missing data with 0
# - Normalize item by item

# - Rating matrix A

# In[ ]:


A = import_data_to_matrix()


# - Observation matrix Ω
# - Normalize item by item (z-scores)
# $$A_{ij} = \frac{A_{ij} - \overline{A{j}}}{std(A_{j})}$$
# 
# Important: 
# - Only observed entries are updated
# - Mean and std is computed only over observed entries

# In[ ]:


W = (A > 0).astype(int)
norm_A, mean_A, stddev_A = zscore_masked_items(A, W)


# ## SVD & ALS
# - SVD with $k = 3$
# - Use result of SVD to initialize ALS (UV)
# - Run ALS (UV) with $k = 3$ and $λ = 0.1$ for 20 iterations

# ### SVD & ALS initialization
# $$A_{nxm} = U_{nxk} V_{kxm}$$

# In[ ]:


# SVD decomposition
U, s, Vt = np.linalg.svd(norm_A, full_matrices=False)
# Using the top k eigenvalues
S = np.zeros((NUMBER_OF_MOVIES , NUMBER_OF_MOVIES))
S[:k, :k] = np.diag(s[:k])
# Initialize ALS with SVD result
# Only first k columns the rest are all set to zero
U = U.dot(S)[:,:k]
V = S.dot(Vt)[:k]


# ### Alternating Least Square
# - Always converge but there is no guarantee that it will converge to the optimal

# - Objective function:
# $$l(U, V) = \frac{1}{2}||Π_{Ω}(A - UV)||_{F}^{2} + \frac{λ}{2}(||U||_{F}^{2} + ||V||_{F}^{2})$$

# In[ ]:


def loss(A, U, V, W, l):
    return ((1/2) * np.sum((W * (A - np.dot(U, V)) ** 2))
            + (l/2) * (np.sum(U ** 2) + np.sum(V ** 2)))


# $$ v_j^* = (\sum_{i}^n {ω_{ij}u_iu_i^T + λI})^{-1} (\sum_{i}^n ω_{ij}a_{ij}u_{i})$$
# 
# $$ u_i^* = (\sum_{j}^m {ω_{ij}v_jv_j^T + λI})^{-1} (\sum_{j}^m ω_{ij}a_{ij}v_{j})$$
# 
# - Trick: Solve a system of linear equations instead of finding the inverse
# 
# $$ (\sum_{i}^n {ω_{ij}u_iu_i^T + λI})v_j^* = (\sum_{i}^n ω_{ij}a_{ij}u_{i})$$
# 
# $$ (\sum_{j}^m {ω_{ij}v_jv_j^T + λI})u_i^* = (\sum_{j}^m ω_{ij}a_{ij}v_{j})$$
# 
# - Note that $u_{i}$ is the $i^{th}$ row of $U$ and $v_{j}$ is the $j^{th}$ column of $V$.

# In[ ]:


for epoch in tqdm(range(iterations)):
    for j, Wj  in enumerate(W.T):
        V[:,j] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(Wj), U)) + lambda_ * np.eye(k),
                                 np.dot(U.T, np.dot(np.diag(Wj), norm_A[:, j])))

    print("Loss l(U,V) after solving for V:", loss(norm_A, U, V, W, lambda_))

    for i, Wi  in enumerate(W):
        U[i] = np.linalg.solve(np.dot(V, np.dot(np.diag(Wi), V.T)) + lambda_ * np.eye(k),
                               np.dot(V, np.dot(np.diag(Wi), norm_A[i].T))).T

    print("Loss l(U,V) after solving for U:", loss(norm_A, U, V, W, lambda_))


# - Reconstruct data from the result of ALS (UV) after 20 iterations.

# In[ ]:


rec_A = np.dot(U, V)
#undo normalization
for j in range(1000):
    rec_A[:,j] *= stddev_A[j]
    rec_A[:,j] += mean_A[j]


# ## Export Predictions
# 

# In[ ]:


extract_submission(rec_A, file="baseline")

