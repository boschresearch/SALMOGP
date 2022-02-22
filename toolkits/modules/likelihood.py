"""
// Copyright (c) 2022 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import tensorflow as tf
import gpflow

from gpflow.likelihoods import ScalarLikelihood
from gpflow import logdensities
from gpflow.base import Parameter
from gpflow.utilities import positive
from gpflow.likelihoods.utils import inv_probit

"""
The following class is adapted from GPflow 2.1.4
(https://github.com/GPflow/GPflow/blob/develop/gpflow/likelihoods/scalar_continuous.py 
Copyright 2017-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).

It is modified to allow multiple noise variances & allow nan values for Y.
"""

class MultiGaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance=np.array([1.0]), variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)
        if not hasattr(variance, "__len__"):
            raise ValueError(
                "The variance need to be an array. Consider using gpflow.likelihoods.Gaussian for scalar variance."
                )
        if any(var <= variance_lower_bound for var in variance):
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))
    
    def validate_matrix(self, Y):
        r"""
        return Y_modified, valid_entries
        
        where Y_modified = Y, but all nan are replaced by 0
        valid_entries is a bool matrix with the same shape as Y with element == True when Y != nan
        """
        
        valid_entries = tf.math.is_finite(Y)
        Y = tf.where(valid_entries, Y, tf.zeros_like(Y))  # required to work with nan in Y
        
        return Y, valid_entries
    
    def _scalar_log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.concat(
            [tf.fill([*tf.shape(F)[:-1], 1], var) for var in self.variance.numpy()],
            axis = -1
            )
        
    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):
        Y, valid_entries = self.validate_matrix(Y)
        logdens = logdensities.gaussian(Y, Fmu, Fvar + self.variance)
        logdens = tf.where(valid_entries, logdens, tf.zeros_like(logdens))
        return tf.reduce_sum(logdens, axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, valid_entries = self.validate_matrix(Y)
        
        ve = - 0.5 * np.log(2 * np.pi)
        ve -= 0.5 * tf.math.log(self.variance)
        ve -= 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance
        
        ve = tf.where(valid_entries, ve, tf.zeros_like(ve))
        return tf.reduce_sum(ve, axis=-1)
