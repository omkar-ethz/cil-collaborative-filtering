import numpy as np
import pandas as pd

number_of_users = 10000
number_of_movies = 1000
k = 3  # number of top eigenvalues of SVD, choose by cross validation
lambda_ = 0.1

data_pd = pd.read_csv('data_train.csv')
users, movies = \
    [np.squeeze(arr) for arr in \
     np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
predictions = data_pd.Prediction.values

# Create data matrix
data = np.full((number_of_users, number_of_movies), np.nan)
for user, movie, pred in zip(users, movies, predictions):
    data[user][movie] = pred

data_test = pd.read_csv('sampleSubmission.csv')
test_users, test_movies = \
    [np.squeeze(arr) for arr in np.split(data_test.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]

from fancyimpute import SoftImpute, BiScaler
solver = SoftImpute(max_rank=3, max_iters=10, n_power_iterations=100)
scaler = BiScaler(min_value=1, max_value=5)
data_normalized = scaler.fit_transform(data)
prediction_norm = solver.fit_transform(data_normalized)
prediction = scaler.inverse_transform(prediction_norm)

with open('submission.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for (user, movie) in zip(test_users, test_movies):
        if abs(np.round(prediction[user][movie]) - prediction[user][movie]) < 0.051:
            ans = int(np.round(prediction[user][movie]))
        else:
            ans = prediction[user][movie]
        f.write("r{}_c{},{}\n".format(user + 1, movie + 1, ans))

