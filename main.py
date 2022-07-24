import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from utils import extract_for_ensemble, create_matrix_from_raw, RAND_SEED
from models import IRSVD, Baseline, GBias, SVP, SVT, RSVD, SVD
from ensemble import Ensemble
from utils import extract_submission

def generate_ensembles():
    data_pd = pd.read_csv("./data/data_train.csv")
    kf = KFold(n_splits=10, shuffle=True, random_state=RAND_SEED)
    # Check whether we have the same splits
    for train_set, test_set in kf.split(data_pd):
        print(train_set)
        print(test_set)
        
    # IRSVD
    params = (
        ("mean", 325, 0.01, 0.02, 0.05, 15),
    )

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            biases, features, eta, lambda1, lambda2, epochs = param
            fname = "irsvd_"+biases+"_"+str(features)+"_"+str(epochs)
            print(param)
            model = IRSVD(train_matrix, biases=biases, features=features,
                        eta=eta, lambda1=lambda1, lambda2=lambda2, epochs=epochs)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)

    params = (
        ("mean", 325, 0.01, 0.02, 0.05, 15),
    )

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        biases, features, eta, lambda1, lambda2, epochs = param
        fname = "irsvd_"+biases+"_"+str(features)+"_"+str(epochs)
        print(param)
        model = IRSVD(train_matrix, biases=biases, features=features,
                        eta=eta, lambda1=lambda1, lambda2=lambda2, epochs=epochs)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)
    
    # ALS
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
    
    # Global biases
    params = (
        (0.001, 5),
    )

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            lambda1, epochs = param
            fname = "global_"+str(epochs)
            print(param)
            model = GBias(train_matrix, lambda1=lambda1, epochs=epochs)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)
    
    params = (
        (0.001, 5),
    )

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        lambda1, epochs = param
        fname = "global_"+str(epochs)
        print(param)
        model = GBias(train_matrix, lambda1=lambda1, epochs=epochs)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)
    
    # SVProjection
    params = (
        (5, 3, 10),
    )

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            eta, K, epochs = param
            fname = "svp_"+str(eta)+"_"+str(K)+"_"+str(epochs)
            print(param)
            model = SVP(train_matrix, eta=eta, K=K, epochs=epochs)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)
    
    params = (
        (5, 3, 10),
    )

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        eta, K, epochs = param
        fname = "svp_"+str(eta)+"_"+str(K)+"_"+str(epochs)
        print(param)
        model = SVP(train_matrix, eta=eta, K=K, epochs=epochs)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)
    
    # SVT
    params = (
        (1.2, 800, 12),
    )

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            eta, tau, epochs = param
            fname = "svt_"+str(eta)+"_"+str(tau)+"_"+str(epochs)
            print(param)
            model = SVT(train_matrix, eta=eta, tau=tau, epochs=epochs)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)
    
    params = (
        (1.2, 800, 12),
    )

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        eta, tau, epochs = param
        fname = "svt_"+str(eta)+"_"+str(tau)+"_"+str(epochs)
        print(param)
        model = SVT(train_matrix, eta=eta, tau=tau, epochs=epochs)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)
    
    # SVD
    params = [8,]

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            k = param
            fname = "svd_"+str(k)
            print(param)
            model = SVD(train_matrix, K=k)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)

    params = [8,]

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        k = param
        fname = "svd_"+str(k)
        print(param)
        model = SVD(train_matrix, K=k)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)
    
    params = (
        (325, 0.01, 0.02, 15),
    )

    for idx, (train_set, test_set) in enumerate(kf.split(data_pd)):
        train_data = data_pd.iloc[train_set]
        test_data = data_pd.iloc[test_set]
        
        train_matrix = create_matrix_from_raw(train_data)
        test_matrix = create_matrix_from_raw(test_data)
        
        for param in params:
            features, eta, lambda1, epochs = param
            fname = "rsvd_"+str(features)+"_"+str(epochs)
            print(param)
            model = RSVD(train_matrix, features=features,
                        eta=eta, lambda1=lambda1, epochs=epochs)
            print(model.train(test_matrix=test_matrix))
            rec_matrix = model.reconstruct_matrix()
            extract_for_ensemble(rec_matrix, fname, idx+1, train=True)
            
    params = (
        (325, 0.01, 0.02, 15),
    )

    train_matrix = create_matrix_from_raw(data_pd)
    for param in params:
        features, eta, lambda1, epochs = param
        fname = "rsvd_"+str(features)+"_"+str(epochs)
        print(param)
        model = RSVD(train_matrix, features=features,
                        eta=eta, lambda1=lambda1, epochs=epochs)
        print(model.train())
        rec_matrix = model.reconstruct_matrix()
        extract_for_ensemble(rec_matrix, fname, 0, train=False)

if __name__ == "__main__":
    generate_ensembles()
    
    model = Ensemble()
    model.train()
    rec_A = model.reconstruct_matrix()
    
    rec_A[rec_A>5] = 5
    rec_A[rec_A<1] = 1
    
    extract_submission(rec_A, file="ensemble")
