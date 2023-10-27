# aleatoric_epistemic_discrimination
This repository contains code and data to reproduce the experiments in our paper: Aleatoric and Epistemic Discrimination: Fundamental Limits of Fairness Interventions.

## Datasets
Datasets are stored in the `data/` directory. It contains UCI-Adult, COMPAS, HSLS, German Credit, and synthetic datasets. In particular, the HSLS dataset has multi-class labels and multiple protected groups.

## FairFront
The `FairFront/` directory contains our implementation of the our main concept---fairness Pareto frontier (**FairFront**). We use it to measure aleatoric discrimination and quantify epistemic discrimination by comparing a classifier's performance to the **FairFront**. This effort converts a functional optimization problem into a convex program with a small number of variables. However, this convex program may involve infinitely many constraints. Hence, we introduce a greedy improvement algorithm that iteratively refines the approximation of **FairFront** and also establish a convergence guarantee.

The main algorithm is presented in `FairFront.py`. For loading different datasets, pre-processing, and applying the **FairFront** method, please see the notebooks in this folder. 

## Benchmarks
We consider five existing fairness-intervention algorithms: **Reduction** (Agarwal et al., 2018),  **EqOdds** (Hardt et al., 2016), **CalEqOdds** (Pleiss et al., 2017), **LevEqOpp** (Chzhen et al., 2019), and **FairProjection** (Alghamdi et al., 2022). Among them, **Reduction** is an in-processing method and the rest are all post-processing methods. 

- This `benchmarks/` directory contains code for comparing benchmark methods, including the original classifier and with the five fairness-intervention algorithms mentioned above. 
- Run `run_benchmark.py` to run benchmark methods. Example: `python3 run_benchmark.py -f caleqodds -i compas -m rf -n 10`.
  - Options:
    - [model name]: gbm, logit, rf (Default: rf)
    - [fair method]: reduction, eqodds, roc, leveraging, original (Default: reduction)
    - [constraint]: eo, sp, (Default: eo)
    - [num iter]: Any positive integer (Default: 10)
    - [inputfile]: adult, compas, german_credit ...  (Default: adult)
    - [seed]: Any integer (Default: 42)
- Use `run-mp-sensitive.py` to run Fair Projection, which uses `utils_with_sensitive`, `coreMP`, and `GroupFair` modules. Example: `python3 run-mp-sensitive.py -i compas -r 10`.
  - Options:
    - [inputfile]: adult, compas, german_credit (Default: adult)
    - [reptition]: number of iterations to run for each point (Default: 10)
- To run the multi-class multi-label Fair Projection on the HSLS dataset, please navigate to the **`multi-group-multi-class-hsls/`** directory first. Example: `python3 run-mp-hsls.py -r 10`



## Plotting
The `plot_helper/` directory contains pickle file of results from our experiments using various benchmark methods, as well as a notebook example to replicate graphs in our paper. 
