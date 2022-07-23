# Collaborative Filtering: Picnic of Ensembled Matrix Factorization Models

This project is part of the Computational Intelligence Lab course (2022) at ETH.

Project [Collaborative Filtering](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/overview).

Team: Picnic

| Name  | Email |
| ------------- | ------------- |
| Andreas Tsouloupas | atsouloupas@student.ethz.ch  |
| Arad Mohammadi | amohammad@student.ethz.ch  |
| Hsiu-Chi Cheng | chengs@student.ethz.ch  |
| Omkar Zade  | omzade@student.ethz.ch  |



## Project Structure Tree

    .
    ├── data                               # should contain following files and folders
    │   ├── ensemble                       # files produced during main.py execution for ensemble of models
    │   │   ├── final
    │   │   └── train
    │   │       ├── 1
    │   │       ├── 2
    │   │       ├── 3
    │   │       ├── 4
    │   │       ├── 5
    │   │       ├── 6
    │   │       ├── 7
    │   │       ├── 8
    │   │       ├── 9
    │   │       └── 10
    │   ├── submissions
    │   ├── data_train.csv
    │   └── sampleSubmission.csv
    ├── experiments_out                    # graphs from experiments
    ├── report
    │   └── report.pdf                     # CIL report
    ├── requirements.txt
    ├── ensemble.py
    ├── main.py                            # Reproduce submission
    ├── models.py
    ├── utils.py
    └── README.md
    

## Getting Started

Setup python environment:

 ```
 # create environment (tested with python 3.10.5)
 python -m venv "cil"
 
 # activate environment 
 source cil/bin/activate

 # install dependencies 
 pip install -r requirements.txt
 
  ```
  
To replicate final submission: (output submission file in folder submissions)
- Warning: this will take a long time (1 day or more)
- Make sure that the project structure is the same as above before executing the command
```
python main.py
```

To replicate individual methods: (output submission file in folder submissions)
- Run entire corresponding jupyter notebook
    - [Singular Value Decomposition (SVD) Baseline1](./svd.ipynb)
    - [Alternating Least Squares (ALS) Baseline2](./baseline.ipynb)
    - [Global Bias (GBias)](./global.ipynb)
    - [Singular Value Projection (SVP)](./svp.ipynb)
    - [Singular Value Thresholding (SVT)](./svt.ipynb)
    - [Reqularized SVD (RSVD)](./rsvd.ipynb)
    - [Improved Reqularized SVD (IRSVD)](./irsvd.ipynb)

To replicate experiments:
Run (all) the jupyter notebook [experiments.ipynb](./experiments.ipynb) which also performs cross-validation

## Documentation (Report)
[Report](./report/report_cil.pdf)
