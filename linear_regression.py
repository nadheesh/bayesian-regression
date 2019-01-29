"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Demonstrate the definition of Linear regression model using PyMC3.
"""

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from theano import shared, tensor as tt

size = 100  # no of data samples
true_intercept = 0.5  # intercept of the regression line
true_slope = 2  # slop (coefficient) of x

x = np.linspace(0, 1, size)  # generate some random values for x in the given range (0,1)

# performs simple linear regression
# y = a + b*x
true_regression_line = true_intercept + true_slope * x

# we can't use true regression values for training the model
# therefore, add some random noise
y = true_regression_line + np.random.normal(scale=.3, size=size)

# split the data points into train and test split
x_train, x_test, y_train, y_test, true_y1, true_y2 = train_test_split(x, y, true_regression_line)

# we use a shared variable from theano to feed the x values into the model
# this is need for PPC
# when using the model for predictions we can set this shared variable to x_test
shared_x = shared(x_train)

# training the model
# model specifications in PyMC3 are wrapped in a with-statement
with pm.Model() as model:
    # Define priors
    x_coeff = pm.Normal('x', 0, sd=20)  # prior for coefficient of x
    intercept = pm.Normal('Intercept', 0, sd=20)  # prior for the intercept
    sigma = pm.HalfCauchy('sigma', beta=10)  # prior for the error term of due to the noise

    mu = intercept + tt.dot(shared_x, x_coeff)  # represent the linear regression relationship

    # Define likelihood
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y_train)

    # Inference!
    trace = pm.sample(1000, njobs=1)  # draw 3000 posterior samples using NUTS sampling

# predicting the unseen y values
# uses posterior predictive checks (PPC)
shared_x.set_value(x_test)  # let's set the shared x to the test dataset
ppc = pm.sample_ppc(trace, model=model, samples=1000)  # performs PPC
predictions = ppc['y'].mean(axis=0)  # compute the mean of the samples draws from each new y

# now you can measure the error
print("\n MSE of simple linear regression using bayesian : {0}\n".format(mean_squared_error(y_test, predictions)))

# plot the traceplot
pm.traceplot(trace)

# let's plot the regression lines
fig = plt.figure(figsize=(5, 5))

pm.plot_posterior_predictive_glm(trace, samples=100,
                                 label='posterior predictive regression lines')
plt.plot(x_test, true_y2, label='true regression line', lw=2., c='g')

plt.plot(x_test, true_y2, 'x', label='true y', color="blue")
plt.plot(x_test, predictions, 'x', label='prediction', color="red")
plt.title('n = {0}'.format(size))
plt.legend(loc=0)
plt.xlabel('x')
plt.ylabel('y')

plt.show()
