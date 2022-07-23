# Collaborative Filtering

This project is part of the Computational Intelligence Lab course (2022) at ETH.

The project website can be found [here](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/overview).

Team: Picnic

| Name  | Email |
| ------------- | ------------- |
| Andreas Tsouloupas | atsouloupas@student.ethz.ch  |
| Arad Mohammadi | amohammad@student.ethz.ch  |
| Hsiu-Chi Cheng | chengs@student.ethz.ch  |
| Omkar Zade  | omzade@student.ethz.ch  |



## Project structure

    .
    ├── data                               # should contain files
        ├── ensemble
            ├── final
            ├── train
                ├── 1
                ├── 2
                ├── 3
                ├── 4
                ├── 5
                ├── 6
                ├── 7
                ├── 8
                ├── 9
                ├── 10
        ├── submissions
        data_train.csv  sampleSubmission.csv
    ├── experiments_out                    # graphs from experiments
    ├── report                              
        ├── report.pdf                     # Final report
    ├── requirements.txt
    ├── ensemble.py
    ├── main.py
    ├── models.py
    ├── utils.py
    └── README.md
    

## Getting Started

To run locally:

 ```
 # create environment
 python -m venv "cil"
 
 # activate environment 
 source cil/bin/activate

 # install dependencies 
 pip install --user -r requirements.txt
 
  ```
  
To replicate final submission: (submission file in folder submissions)
- Warning: this will take some time
- Make sure that the project structure is satisfied
```
python main.py # 
```

To replicate individual methods: (submission file in folder submissions)
- Run entire corresponding jupyter notebook
    - [Singular Value Decomposition (SVD) Baseline1](./svd.ipynb)
    - [Alternating Least Squares (ALS) Baseline2](./baseline.ipynb)
    - [Global Bias (GBias)](./global.ipynb)
    - [Singular Value Projection (SVP)](./svp.ipynb)
    - [Singular Value Thresholding (SVT)](./svt.ipynb)
    - [Reqularized SVD (RSVD)](./rsvd.ipynb)
    - [Improved Reqularized SVD (RSVD)](./irsvd.ipynb)

To replicate experiments:
Run (all) the jupyter notebook [experiments.ipynb](./experiments.ipynb) which also performs cross-validation

## Paper Report
[Report](./report/report.pdf)