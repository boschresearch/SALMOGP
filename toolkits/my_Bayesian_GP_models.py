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
import matplotlib.pyplot as plt

from tensorflow_probability import distributions as tfd
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.ci_utils import ci_niter

from os import path
import sys
sys.path.append(path.dirname(__file__))

from my_ALGP_models import myMOGPR
from modules.query import acquisitioner, safety_manager

f64 = gpflow.utilities.to_default_float

free_mem = 20 # in GB
MEMORY_LIMIT = (free_mem - 5) * 2**30

"""
Some pieces of the following class BayesianMOGPR is adapted from GPflow 2.1.4
(https://github.com/GPflow/GPflow/blob/develop/doc/source/notebooks/advanced/mcmc.pct.py
Copyright 2016-2021 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""

class BayesianMOGPR(myMOGPR):
    def __init__(
        self,
        data,
        kernel,
        noise_variance = 1.0,
        mean_function = None,
        num_latent_gps: int = None,
        thin_steps: int=20,
        num_burnin_steps: int=300,
        num_samples: int=100,
        history_initialize: bool = True,
        data_args=None
    ):
        super().__init__(
            data, kernel, noise_variance=noise_variance,
            mean_function=mean_function, num_latent_gps=num_latent_gps,
            history_initialize=history_initialize, data_args=data_args
            )
        
        self.assign_prior()
        
        self.initialize_inference_state()
        
        self.num_burnin_steps = ci_niter(num_burnin_steps)
        self.thin_steps = ci_niter(thin_steps)
        self.num_samples = ci_niter(num_samples)
        
        self.set_training_args(opt='Adam', MAXITER=500, learning_rate=0.1)
        self.inference_mode = 'bayesian' # or 'max_likelihood'
    
    def set_inference_mode(self, bayesian:bool=True):
        if bayesian:
            self.inference_mode = 'bayesian'
        else:
            self.inference_mode = 'max_likelihood'


    def return_config(self):
        output = {}
        
        output['num_burnin_steps'] = self.num_burnin_steps
        output['thin_steps'] = self.thin_steps
        output['num_samples'] = self.num_samples
        
        return output
        
    def assign_prior(self):
        # prior to kernel
        if isinstance(self.kernel, MultioutputKernel):
            for i in range(self.num_latent_gps):
                self.kernel.kernels[i].lengthscales.prior = tfd.Gamma(f64(1.5), f64(1.0))
                self.kernel.kernels[i].variance.prior = tfd.Gamma(f64(2.5), f64(1.0))
                
            if hasattr(self.kernel, 'W'):
                self.kernel.W.prior = tfd.Normal(f64(0.0), f64(2.0))
        
        else:
            self.kernel.lengthscales.prior = tfd.Gamma(f64(1.5), f64(1.0))
            self.kernel.variance.prior = tfd.Gamma(f64(2.5), f64(1.0))
        
        self.likelihood.variance.prior = tfd.Gamma(f64(1.5), f64(3.0))
        
        return True
    
    def set_training_args(
            self,
            opt: str='Adam',
            **kwargs
            ):
        self.opt = opt
        self.opt_kwargs = kwargs
    
    def initialize_first_sample(self):
        for parameter in self.trainable_parameters:
            new_value = parameter.prior.sample(tf.shape(parameter))
            parameter.assign(new_value)
    
    def HMC_sampling(
            self,
            ):
        timer = -time.perf_counter()
        
        self.initialize_inference_state()
        self.training(opt=self.opt, convergence_check=False, **self.opt_kwargs)
        
        # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
        self.hmc_helper = gpflow.optimizers.SamplingHelper(
            self.log_posterior_density, self.trainable_parameters
            )
        
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.hmc_helper.target_log_prob_fn,
            num_leapfrog_steps=10, step_size=0.01
            # move {num_leapfrog_steps} * {step_size} in one HMC step
            )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc,
            num_adaptation_steps=int(0.3*self.num_burnin_steps),
            target_accept_prob=f64(0.75),
            adaptation_rate=0.1
            )
        
        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain( # in total {num_results + num_burnin_steps} samples, but first {num_burnin_steps} won't be used
                num_results=self.num_samples,
                num_burnin_steps=self.num_burnin_steps, # warm_up steps
                num_steps_between_results=self.thin_steps,
                current_state=self.hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                )
        
        inference_success = False
        while not inference_success:
            self.initialize_first_sample()
            try:
                with tf.device('/cpu:0'):
                    self.samples, self.trace = run_chain_fn()
                inference_success = True
            except:
                print("ERROR in MCMC sampling - repeat")
        
        
        timer += time.perf_counter()
        
        self.parameter_samples = self.hmc_helper.convert_to_constrained_values(self.samples)
        
        return timer
    
    def return_all_model_samples(self):
        param_to_name = {
            param: name for name, param in gpflow.utilities.parameter_dict(self).items()
            }
        
        unconstrained_dict = {
            param_to_name[param]: val for val, param in zip(self.samples, self.trainable_parameters)
            }
        constrained_dict = {
            param_to_name[param]: val for val, param in zip(self.parameter_samples, self.trainable_parameters)
            }
        
        return unconstrained_dict, constrained_dict
    
    def predict_f_all_models(
            self,
            X_test,
            full_cov: bool = False,
            full_output_cov: bool = False
            ):
        
        self.initialize_inference_state(
            X_test=X_test,
            full_cov=full_cov,
            full_output_cov=full_output_cov
            )
        
        
        for i in range(0, self.num_samples):
            for var, var_samples in zip(self.hmc_helper.current_state, self.samples):
                var.assign(var_samples[i])
            mu, var = super().predict_f(X_test, full_cov=full_cov, full_output_cov=full_output_cov, mem_limit = MEMORY_LIMIT)
            
            if self.inference_result['mu_all'] is None:
                self.inference_result['mu_all'] = mu[None,...]
            else:
                self.inference_result['mu_all'] = tf.concat(
                    (self.inference_result['mu_all'], mu[None,...]),
                    axis=0
                    )
            
            if self.inference_result['var_all'] is None:
                self.inference_result['var_all'] = var[None,...]
            else:
                self.inference_result['var_all'] = tf.concat(
                    (self.inference_result['var_all'], var[None,...]),
                    axis=0
                    )
        
        return True
            
            
    def initialize_inference_state(
            self, X_test=None,
            mu_all=None,
            var_all=None,
            full_cov=False,
            full_output_cov=False
            ):
        
        self.inference_result = {
            #'state': False,
            'X_test': X_test,
            'mu_all': mu_all,
            'var_all': var_all,
            'full_cov': full_cov,
            'full_output_cov': full_output_cov
            }
    
    
    def predict_f(self, X_test, full_cov: bool = False, full_output_cov: bool = False):
        """
        psudo code
        1. check if self.X_test == X_test
           if yes, check if self.cov_all.numpy().shape is as expected
           if yes, skip 2 (this step basically calls the result if exist, it is faster)
        2. predict with all sets of hyperpar.s & save (sampling should already be done)
        3. use the prediction results for Gaussian mix (moment matching)
        """
        if self.inference_mode == 'max_likelihood':
            return super().predict_f(
                X_test,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                mem_limit = MEMORY_LIMIT
                )
        while True:
            if np.array_equal(self.inference_result['X_test'], X_test):
                if self.inference_result['full_cov'] == full_cov and self.inference_result['full_output_cov'] == full_output_cov:
                    extract_var = self.inference_result['var_all']
                    break
                elif not self.inference_result['full_cov'] and self.inference_result['full_output_cov'] and not full_cov and not full_output_cov:
                    # [self.num_samples, N, P, P] -> [self.num_samples, N, P]
                    extract_var = tf.linalg.diag_part(self.inference_result['var_all'])
                    break
                elif self.inference_result['full_cov'] and not self.inference_result['full_output_cov'] and not full_cov and not full_output_cov:
                    # [self.num_samples, P, N, N] -> [self.num_samples, N, P]
                    extract_var = tf.transpose(
                            tf.linalg.diag_part(self.inference_result['var_all']),
                            [0, 2, 1])
                    break
                elif self.inference_result['full_cov'] and self.inference_result['full_output_cov']:
                    # [self.num_samples, N, P, N, P]
                    if full_cov and not full_output_cov:
                        # -> [self.num_samples, P, N, N]
                        extract_var = tf.transpose(
                            tf.linalg.diag_part(
                                tf.transpose(self.inference_result['var_all'], [0,1,3,2,4])
                                # [self.num_samples, N, N, P, P]
                                ), # [self.num_samples, N, N, P]
                            [0, 3, 1, 2]) # [self.num_samples, P, N, N]
                        break
                    if not full_cov and full_output_cov:
                        # -> [self.num_samples, N, P, P]
                        extract_var = tf.transpose(
                            tf.linalg.diag_part(
                                tf.transpose(self.inference_result['var_all'], [0,2,4,1,3])
                                # [self.num_samples, P, P, N, N]
                                ), # [self.num_samples, P, P, N]
                            [0, 3, 1, 2]) # [self.num_samples, N, P, P]
                        break
                    else:
                        # -> [self.num_samples, N, P]
                        extract_var = tf.linalg.diag_part(
                            tf.transpose(
                                tf.linalg.diag_part(
                                    tf.transpose(self.inference_result['var_all'], [0,2,4,1,3])
                                    # [self.num_samples, P, P, N, N]
                                    ), # [self.num_samples, P, P, N]
                                [0, 3, 1, 2]) # [self.num_samples, N, P, P]
                            ) # [self.num_samples, N, P]
                        break
                
            self.predict_f_all_models(X_test, full_cov=full_cov, full_output_cov=full_output_cov)
            extract_var = self.inference_result['var_all']
            break
        
        mu_mean = tf.reduce_mean(self.inference_result['mu_all'], axis=0)
        
        if not full_output_cov:
            if full_cov:
                #[P, N, N]
                raise ValueError('not finished yet')
            else:
                # [N, P]
                bbT = tf.square(mu_mean)
                aaT = tf.reduce_mean(
                    tf.square(self.inference_result['mu_all']),
                    axis=0)
        else:
            # [N, P, P]
            bbT = tf.expand_dims(mu_mean, 1) \
                * tf.expand_dims(mu_mean, 2)
            aaT = tf.reduce_mean(
                    tf.expand_dims(self.inference_result['mu_all'], 2) \
                  * tf.expand_dims(self.inference_result['mu_all'], 3),
                axis = 0
                )
            if full_cov:
                # [N, P, N, P]
                raise ValueError('not finished yet')
        var_mean = tf.reduce_mean(extract_var, axis=0) + aaT - bbT
        
        return mu_mean, var_mean
    
    
    def update_dataset(
        self, data_new, data_args = None
    ):
        super().update_dataset(data_new, data_args)
        self.initialize_inference_state()
        



class ALBMOGPR(BayesianMOGPR, acquisitioner):
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
            ):
        
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
        # be aware that myVGP.update_dataset will also call self.history_update
        if return_index:
            return return_D, return_arg
        return return_D
    

class SafetyBMOGPR(BayesianMOGPR, safety_manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)














