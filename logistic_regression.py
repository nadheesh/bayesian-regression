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
"""

import numpy as np
import pymc3 as pm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from theano import shared, tensor as tt


def invlogit(x):
    """
    sigmoid operator
    :param x: int/float or array like data-structure
    :return: sigmoid of x
    """
    if type(x) is np.ndarray:
        return np.exp(x) / (1 + np.exp(x))
    return tt.exp(x) / (1 + tt.exp(x))


size = 100  # no of data samples
true_intercept = 0.5  # intercept of the regression line
true_slope = 2  # slop (coefficient) of x

x = np.linspace(0, 1, size)  # generate some random values for x in the given range (0,1)

# performs simple linear regression
# y = sigmoid(a + b*x)
true_regression_line = invlogit(true_intercept + true_slope * x)

# we can't use true regression values for training the model
# therefore, add some random noise
y = true_regression_line + np.random.normal(scale=.3, size=size)

# assume this is a binary classifications
# convert y values to labels
true_regression_line = true_regression_line >= 0.5
y = y >= 0.5

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

    reg = intercept + tt.dot(shared_x, x_coeff)
    p = pm.Deterministic("p", invlogit(reg))  # represent the logistic regression relationship

    # Define likelihood
    likelihood = pm.Bernoulli('y', p=p, observed=y_train)

    # Inference!
    trace = pm.sample(1000)  # draw 3000 posterior samples using NUTS sampling

# predicting the unseen y values
# uses posterior predictive checks (PPC)
shared_x.set_value(x_test)  # let's set the shared x to the test dataset
ppc = pm.sample_ppc(trace, model=model, samples=1000)  # performs PPC
predictions = ppc['y'].mean(axis=0)  # compute the mean of the samples draws from each new y

predictions = predictions >= 0.5
# now you can check the error
print("Accuracy of logistic regression using bayesian : {0}".format(accuracy_score(y_test, predictions)))

# plot the traceplot
pm.traceplot(trace)

# TODO add the graphs
# let's plot the regression lines
# fig = plt.figure(figsize=(5, 5))
#
# pm.plot_posterior_predictive_glm(trace, samples=100,
#                               label='posterior predictive regression lines')
# plt.plot(x_test, true_y2, label='true regression line', lw=2., c='g')
#
# plt.plot(x_test, y_test, 'x', label='true y', color="blue")
# plt.plot(x_test, predictions, 'x', label='prediction', color="red")
# plt.title('n = {0}'.format(size))
# plt.legend(loc=0)
# plt.xlabel('x')
# plt.ylabel('y')
#
# plt.show()
