
# Bayesian Regression

A simple demonstration of the Bayesian Regression models using PyMC3.

## Bayesian Linear and Logistic regression

Demonstrates the implementations of linear regression models based on Bayesian inference.

## Bayesian GP-Regression 

GP regression with ARD. 

Can select between the MAP inference and MCMC sampling. 

To select MAP inference initiate the model as follows.

    model = BayesianGPRegression(is_MAP=True)

To select MCMC sampling set `is_MAP` to False during the model initiation. 

The feature relevance can be retrieved as follows.

    ard_scores = model.ard_coefficients()

Notice that if `is_MAP` is set to `True` then this method will return only the scores. However, if MCMC sampling used, then the uncertainty of the ARD scores will be returned with the scores as follows.

    model = BayesianGPRegression(is_MAP=False)
    ard_scores, ard_uncertainty = model.ard_coefficients()
