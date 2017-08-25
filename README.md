# Bayesian Learning
This repository contains the implementation of a Bayesian Network for modelling probability distributions of several variables pertaining US Universities obtained from [US News and World Report](https://www.usnews.com/education) for the course CSE 574: Introduction to Machine Learning at SUNY Buffalo.

## Problem Description
the goal of the project is to construct a Bayesian Network correlating various features of the US Universities' data with a joint probability distribution, such that it's log likelihood is better than that of data if the variables are assumed to be independent.

## Requirements
```
  1. xlrd==1.0.0

  2. numpy==1.13.0

  3. matplotlib==1.5.1

  4. scipy==0.19.1

  5. pandas==0.18.1
```

## Data Exploration

Before constructing the Bayesian Network, we determine the correlation among the data variables by calculating the covariance matrix and the correlation matrix.

**1. Covariance Matrix**

![Image](https://github.com/darshanbagul/BayesianNetworks/blob/master/results/covarianceMat.png)

**2. Correlation Matrix**
![Image](https://github.com/darshanbagul/BayesianNetworks/blob/master/results/correlationMat.png)

## Implementation

Follow along the **Ipython notebook** for step by step explanation along with the implementation.

## Results

After the detailed analysis, as explained in the Ipython notebook, we constructed the following Bayesian Network as shown in the diagram below. Directed arrows define dependence relationship between the variables.

![Image](https://github.com/darshanbagul/BayesianNetworks/blob/master/results/final_BNGraph.png)
