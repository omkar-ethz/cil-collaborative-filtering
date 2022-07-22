# Collaborative Filtering

This project is part of the Computational Intelligence Lab course (2022) at ETH.

The project website can be found [here](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/overview).

Team: 	PICNIC

| Name  | Email |
| ------------- | ------------- |
| Arad Mohammadi | amohammad@student.ethz.ch  |
| Andreas Tsouloupas | atsouloupas@student.ethz.ch  |
| Hsiu-Chi Cheng | chengs@student.ethz.ch  |
| Omkar Zade  | omzade@student.ethz.ch  |



## Project structure

    .
    ├── data                               # should contain files data_train.csv  sampleSubmission.csv
    ├── experiment_results                 # results from experiments
        ├── graphs                         # directory for saving graphs
        ├── preprocessed                   # directory for saving .csv
        ├── raw                            # directory for saving outputs
    ├── report                              
        ├── report.pdf                     # Final report
    ├── src 
        ├── experiments                    # experiment scripts
    ├── requirements.txt
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
  
To replicate final submission: 
```
python main.py 
```

To replicate experiments:
```
cd src/experiments # You should be in the experiment directory for the experiments to run
python <experiment> 
```

To run cross-validation test:
```
python cross_validation.py --model <model> [--<parameters> <value>]
```

Valid models names are the following: 

```autoencoder```, ```bsgd``` , ```svd_shrinkage```, ```ncf```

## Scientific Report
[Report](./report/report.pdf)