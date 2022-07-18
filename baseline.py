# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from utils import extract_for_ensemble, create_matrix_from_raw, RAND_SEED
from models import Baseline

# %% [markdown]
# ## KFolds
# - Split data into 10 folds (90% train set)
# - Use random state for reproducibility

# %%
data_pd = pd.read_csv("./data/data_train.csv")
kf = KFold(n_splits=10, shuffle=True, random_state=RAND_SEED)
# Check whether we have the same splits
for train_set, test_set in kf.split(data_pd):
    print(train_set)
    print(test_set)

# %% [markdown]
# ## Models
# - Predict matrix for different models and parameters
# - Save the produced matrix for ensemble (for all folds)
# - Also train on entire dataset

# %% [markdown]
# ### Baseline

# %% [markdown]
# - For ensemble training

# %%
params = (
    (3, 0.1, 3),
)

for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
    train_data = data_pd.iloc[train_set]
    test_data = data_pd.iloc[test_set]
    
    train_matrix = create_matrix_from_raw(train_data)
    test_matrix = create_matrix_from_raw(test_data)
    
    for param in params:
        K, lambda1, epochs = param
        fname = "baseline_"+str(K)+"_"+str(epochs)
        print(param)
        model = Baseline(train_matrix, K=K, lambda1=lambda1, epochs=epochs)
        print(model.train(test_matrix=test_matrix))
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, idx+1, train=True)

# %% [markdown]
# - Entire dataset training

# %%
params = (
    (3, 0.1, 3),
)

train_matrix = create_matrix_from_raw(data_pd)
print(data_pd.shape)
for param in params:
    K, lambda1, epochs = param
    fname = "baseline_"+str(K)+"_"+str(epochs)
    print(param)
    model = Baseline(train_matrix, K=K, lambda1=lambda1, epochs=epochs)
    print(model.train())
    rec_matrix = model.reconstruct_matrix()
    extract_for_ensemble(rec_matrix, fname, 0, train=False)


