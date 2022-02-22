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
from typing import Union, Optional, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time

import gpflow
from gpflow.logdensities import multivariate_normal
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.util import data_input_to_tensor

from os import path
import sys
sys.path.append(path.dirname(__file__))

from gpflow.models.training_mixins import InternalDataTrainingLossMixin

from my_toolkit import savefile

from modules.likelihood import MultiGaussian
from modules.model_history import history_manager_dataExclusive, history_manager_dataInclusive
from modules.model_manager import trainer, parameter_manager
from modules.query import acquisitioner, safety_manager

free_mem = 20 # in GB
MEMORY_LIMIT = (free_mem - 2) * 2**30

"""
Some pieces of the following class myMOGPR is adapted from GPflow 2.1.4
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""

class myMOGPR(GPModel, InternalDataTrainingLossMixin, history_manager_dataInclusive, trainer, parameter_manager):
    def __init__(
        self,
        data,
        kernel,
        noise_variance = 1.0,
        mean_function = None,
        num_latent_gps: int = None,
        history_initialize: bool = True,
        data_args=None
    ):
        _, Y_data = data
        
        if hasattr(noise_variance, '__len__'):
            lik = MultiGaussian(np.array(noise_variance))
        else:
            lik = MultiGaussian(np.array([noise_variance for _ in range(np.shape(Y_data)[1])]))
            
        if num_latent_gps is None or isinstance(kernel, MultioutputKernel):
            """
            for MultioutputKernel,
            GPModel.calc_num_latent_gps_from_data(...)
            returns kernel.num_latent_gps
            """
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, lik)
        
        super().__init__(kernel, lik, mean_function, num_latent_gps)
        
        self.data = data_input_to_tensor(data)
        _, Y = self._aggregate_data()
        self.num_data, self.num_task = np.shape(Y)
        if history_initialize:
            self.history_init(data_args)
        
    
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """
        could add penalty here (all training, bayesian inference, etc
                                use down stream functions of this one)
        """
        return self.log_marginal_likelihood()
    
    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance,
        and I is the corresponding identity matrix.
        
        K has shape equal to either [N, N] or [N, P, N, P]
        """
        
        var = self.likelihood.variance
        N = tf.shape(K)[0] # notice that self.num_data may not == N due to _aggregate_data()
        P = self.num_latent_gps
        
        k_reshape = tf.reshape(K, [N*P, N*P])
        
        k_diag = tf.linalg.diag_part(k_reshape)
        s_diag = tf.tile(var, [N])
        Ks = tf.linalg.set_diag(k_reshape, k_diag + s_diag)
        
        return tf.reshape(Ks, tf.shape(K))
            
            
    
    def _squeeze_kernel(self, K, squeeze_part:str = 'full'):
        r"""
        input: K: kernel, shape [N1, P, N2, P]
               squeeze_part: 'full' or 'right'
        
        output: kernel, shape [PN1, PN2] or [N1, P, PN2]
        
        where output[i::P, j::P] = K[:, i, :, j]
        or    output[:,:,j::P] = K[..., j]
        """
        
        N1, P, N2, _ = tf.shape(K) # notice that self.num_data may not == N due to _aggregate_data()
        
        if squeeze_part.lower() == 'full':
            return tf.reshape(K, [N1*P, N2*P])
        elif squeeze_part.lower() == 'right':
            return tf.reshape(K, [N1, P, N2*P])
        else:
            raise ValueError("unknown input")
    
    def _aggregate_data(self):
        X, Y = self.data
        X = np.array(X)
        Y = np.array(Y)
        
        _, D = np.shape(X)
        P = self.num_latent_gps
        
        dtype = X.dtype.descr * D
        struct = X.view(dtype)
        
        _, idx, count = np.unique(struct, return_index=True, return_counts=True)
        
        X_aggr = tf.constant(X[idx].reshape([-1, D]))
        Y_aggr = Y[idx].reshape([-1, P])
        
        for j, c in enumerate(count):
            if c == 1:
                continue
            Yj = Y[np.in1d(struct, struct[idx[j]])]
            
            for p in range(P):
                entry = np.isfinite(Yj[:, p])
                if entry.sum() == 1:
                    Y_aggr[j, p] = Yj[entry, p]
                elif entry.sum() > 1:
                    ele = np.unique(Yj[entry, p])
                    if len(ele) > 1:
                        raise ValueError('ambiguous output, make sure each input only has 1 value for each element of the corresponding output')
                    Y_aggr[j, p] = ele
                    
        Y_aggr = tf.constant(Y_aggr)
        
        return (X_aggr, Y_aggr)
    
    def _return_KNN_with_observed_outputs(self):
        X, Y = self._aggregate_data()
        
        # kernel with likelihood variance(s)
        K = self.kernel(X, full_cov=True, full_output_cov=True) # full_output_cov is default to True anyway, just to avoid confusion
        
        N = tf.shape(K)[0] # notice that self.num_data may not == N due to _aggregate_data()
        P = self.num_latent_gps
        
        #if noise_free:
        #    ks = K
        #else:
        ks = self._add_noise_cov(K)
        
        # re-shaping & dealing with unobserved outputs
        ks = tf.reshape(ks, [N*P, N*P])
        Y = tf.reshape(Y, [-1,1])
        """
        notice: if P==1
        observed_entry == all true
        so ks_observed == ks
        """
        observed_entry = tf.squeeze(tf.math.is_finite(Y))
        ks_observed = tf.boolean_mask(
            tf.boolean_mask(ks
                ,observed_entry, axis=0)
            , observed_entry, axis=1
            )
        Y_observed = tf.boolean_mask(Y, observed_entry, axis=0)
        
        return ks_observed, Y_observed, observed_entry
        
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta) = \log N(Y | m, K)
            = constant - 1/2 \log det(K) - 1/2 (Y-m)^T K^-1 (Y-m).

        """
        X, Y = self._aggregate_data()
        
        ks_observed, Y_observed, observed_entry = self._return_KNN_with_observed_outputs()
        
        m = tf.tile(self.mean_function(X), [1, self.num_latent_gps])
        m_observed = tf.boolean_mask(tf.reshape(m, [-1,1]), observed_entry, axis=0)
        
        
        L = tf.linalg.cholesky(ks_observed)
        
        # [N*P,] log-likelihoods for of Y, all output channels considered
        log_prob = multivariate_normal(Y_observed, m_observed, L)
        
        return tf.reduce_sum(log_prob)
    
    def predict_f(
            self,
            Xnew: InputData,
            full_cov=False,
            full_output_cov=False,
            mem_limit = MEMORY_LIMIT
            ) -> MeanAndVariance:
        r"""
        make sure the kernel is a MultioutputKernel
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        if not isinstance(self.kernel, MultioutputKernel):
            raise ValueError("this method currently only supports MultioutputKernel, make sure the model has the correct kernel object")
        
        X_data, Y_data = self._aggregate_data()
        
        # processed test data, to kind of make too huge Xnew be run in sequence
        needed_size = self.check_kmn_size(np.shape(Xnew)[0])
        
        if full_cov or needed_size <= mem_limit:
            X_test_list = [Xnew]
        else:
            num_needed_inf = int(np.ceil(needed_size / mem_limit))
            idx = np.linspace(0, np.shape(Xnew)[0], num_needed_inf, endpoint=False, dtype=int)
            
            X_test_list = [Xnew[:idx[1], ...]]
            for i in range(1, num_needed_inf-1):
                X_test_list.append(Xnew[idx[i]:idx[i+1], ...])
            X_test_list.append(Xnew[idx[-1]:, ...])
            
        # prepare needed matrix from previous obervations
        kmm, Y_observed, observed_entry = self._return_KNN_with_observed_outputs()
        m = tf.tile(self.mean_function(X_data), [1, self.num_latent_gps])
        m_observed = tf.boolean_mask(tf.reshape(m, [-1,1]), observed_entry, axis=0)
        err = Y_observed - m_observed
        """
        L = cholesky(K) -> K = L @ L.T -> K^-1 = L.T^-1 @ L^-1
        so B.T @ K^-1 @ B = A.T @ A, where A = L^-1 @ B
        """
        mu_list = []
        cov_list = []
        for Xt in X_test_list:
            kmn = self.kernel(X_data, Xt, full_cov=True, full_output_cov=True) # full_output_cov is default to True anyway, just to avoid confusion
            M, P, N, _ = tf.shape(kmn)
            kmn = tf.reshape(kmn, [M*P, N*P])
            kmn = tf.boolean_mask(kmn, observed_entry, axis = 0) # [MP-#unobserved, NP]
            knn = self.kernel(Xt, full_cov=full_cov, full_output_cov=full_output_cov)
            
            # now we have everything, compute predictive mean & cov
            # I don't use gpflow.conditionals.base_condition,
            # because my computation is a 1-D gp while knn has a P-dim shape (full_cov, full_output_cov should be taken care)
            
            # calculating full [N, P, N, P] is impossible as N grows
            # calculate only what we need
            Lm = tf.linalg.cholesky(kmm)
            A = tf.linalg.triangular_solve(Lm, kmn, lower=True) # [MP-#unob, NP]
            
            if not full_cov and not full_output_cov: # return [N, P]
                A_res = tf.reshape(A, [-1, N, P])
                K_Kinv_K = tf.reduce_sum(tf.square(A_res), axis=-3)
            elif not full_cov and full_output_cov: # return [N, P, P]
                A_res = tf.reshape(A, [-1, N, P])
                K_Kinv_K = tf.einsum('...mnp, ...mnq -> ...npq', A_res, A_res)
            elif full_cov and not full_output_cov: # return [P, N, N]
                A_res = tf.reshape(A, [-1, N, P])
                K_Kinv_K = tf.einsum('...mnp, ...mkp -> ...pnk', A_res, A_res)
            else: # return [N, P, N, P]
                K_Kinv_K = tf.matmul(A, A, transpose_a = True) # [NP, NP]
                K_Kinv_K = tf.reshape(K_Kinv_K, [N, P, N, P])
            
            f_var = knn - K_Kinv_K
            
            # compute functional mean
            # note: A = Lm^-1 @ kmn, shape [MP-#unob, NP]
            A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)
            # now A = Lm^H^-1 @ Lm^-1 @ kmn, shape [MP-#unob, NP]
            A = tf.reshape(tf.transpose(A), [N, P, -1])
            f_mean_zero = tf.squeeze(tf.linalg.matmul(A, err), axis=-1)
            
            f_mean = f_mean_zero + self.mean_function(Xt)
            
            mu_list.append(f_mean)
            cov_list.append(f_var)
            
        f_mean = tf.concat(mu_list, axis=0)
        f_var = tf.concat(cov_list, axis=0)
        # full_cov == False means cov_all is [N, P] or [N, P, P]
        # full_cov == True means cov_list has only 1 element
        
        return f_mean, f_var
    
    def check_kmn_size(self, N_test):
        r"""
        N_X_data is the N of observed X after aggregation
        (the corresponding Y may not be fully observed, but we still need full size)

        calculate memory size needed for computing kmn,
        if too huge space needed, we could try doing prediction to the N_test
        points in sequence (time-space trade-off)
        """
        MP = self.num_data * self.num_task
        NP = N_test * self.num_task
        size_double = sys.getsizeof(tf.constant(0, dtype='double'))
        size_kmn = NP * MP * size_double
        # size of Lm^-1 kmn = size_kmn
        return size_kmn
    
    
    def update_dataset(
        self, data_new:RegressionData, data_args = None
    ):
        X_data, Y_data = self.data
        X_new, Y_new = data_new
        
        self.data = data_input_to_tensor(
                (np.vstack((X_data, X_new)), np.vstack((Y_data, Y_new)))
                )
        
        _, Y = self._aggregate_data()
        self.num_data, _ = np.shape(Y)
        
        if data_args is None:
            data_args = [None] * np.shape(X_new)[0]
        self.history_update(data_args)
        
    
    def export_data(self, path):
        X, Y = self.data
        history = self.history["data_history"]
        
        tosave = {
            'X': X.numpy(),
            'Y': Y.numpy(),
            'log': history
            }
        
        return savefile(path, tosave)


class ALMOGPR(myMOGPR, acquisitioner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict_log_density_full_output(self, data):
        X, Y = data
        
        f_mean, f_var = self.predict_f(X)
        
        return gpflow.logdensities.gaussian(Y, f_mean, f_var + self.likelihood.variance)
    
    def query_points(
            self, Dpool,
            full_output_cov: bool=True,
            full_task_query: bool=True,
            num_return: int = 1,
            acquition_function: str='entropy',
            return_index: bool=False,
            original_args=None,
            pool_with_task_indicator:bool=False,
            #data_history_update: bool=False
    ) -> InputData:
        
        Xdata, _ = self._aggregate_data()
        
        timer = -time.perf_counter()
        
        return_D, return_arg = super().query_points(
            Dpool,
            Xdata=Xdata,
            full_output_cov=full_output_cov,
            full_task_query=full_task_query,
            num_return=num_return,
            acquition_function=acquition_function,
            return_index=True,
            original_args=original_args,
            pool_with_task_indicator=pool_with_task_indicator
            )
            
        timer += time.perf_counter()
        
        self.history["point_selection_time"].append(timer)
        """
        if data_history_update:
            self.history_update(return_arg)
        """ # be aware that myVGP.update_dataset will also call self.history_update
        if return_index:
            return return_D, return_arg
        return return_D
    

class SafetyMOGPR(myMOGPR, safety_manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    





