"""
Copyright 2019 Nadheesh Jihan

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

class BayesianGPRegression:

    def __init__(self, is_MAP=True):
        self.is_MAP = is_MAP

    def fit(self, X, y):
        """
        train model
        :param X:
        :param y:
        :return:
        """

        # bayesian matern kernel using gaussian processes
        with pm.Model() as self.model:
            l = pm.Gamma("l", alpha=2, beta=1, shape=X.shape[1])
            nu = pm.HalfCauchy("nu", beta=1)

            cov = nu ** 2 * pm.gp.cov.ExpQuad(X.shape[1], l)

            self.gp = pm.gp.Marginal(cov_func=cov)

            sigma = pm.HalfCauchy("sigma", beta=1)
            y_ = self.gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

            if self.is_MAP:
                self.map_trace = [pm.find_MAP()]

            else:
                self.map_trace = pm.sample(500, tune=500)

    def predict(self, X, name='f_pred', with_error=False):
        """
        predict using the train model
        :param X:
        :return:
        """
        if not hasattr(self, 'model'):
            raise AttributeError("train the model first")

        with self.model:
            f_pred = self.gp.conditional(name, X)
            pred_samples = pm.sample_ppc(self.map_trace, vars=[f_pred], samples=20, random_seed=seed)
            y_pred, error = pred_samples[name].mean(axis=0), pred_samples[name].std(axis=0)

        if with_error:
            return y_pred, error
        return y_pred

    def ard_coefficients(self):

        if self.is_MAP:
            return self.map_trace[0]['l']

        return np.mean(self.map_trace['l'].T, axis=1), np.std(self.map_trace['l'].T, axis=1)
