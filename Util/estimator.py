# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from sklearn.decomposition import PCA
import numpy as np
import itertools
from types import SimpleNamespace

# Standard PCA
class PCAEstimator():
    def __init__(self, n_components):
        self.n_components = n_components
        self.solver = 'full'
        self.transformer = PCA(n_components, svd_solver=self.solver)
        self.batch_support = False

    def get_param_str(self):
        return f"pca-{self.solver}_c{self.n_components}"

    def fit(self, X):
        self.transformer.fit(X)

        # Save variance for later
        self.total_var = X.var(axis=0).sum()

        # Compute projected standard deviations
        self.stdev = np.dot(self.transformer.components_, X.T).std(axis=1)

        # Sort components based on explained variance
        idx = np.argsort(self.stdev)[::-1]
        self.stdev = self.stdev[idx]
        self.transformer.components_[:] = self.transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*self.transformer.components_[[i, j]])
            for (i, j) in itertools.combinations(range(self.n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('IPCA components not orghogonal, max dot', np.abs(dotps).max())

        self.transformer.mean_ = X.mean(axis=0, keepdims=True)

    def get_components(self):
        var_ratio = self.stdev**2 / self.total_var
        return self.transformer.components_, self.stdev, var_ratio

    def proj(self, X):
        return self.transformer.inverse_transform(self.transformer.transform(X))

def get_estimator(name, n_components, alpha):
    if name == 'pca':
        return PCAEstimator(n_components)
    else:
        raise RuntimeError('Unknown estimator')