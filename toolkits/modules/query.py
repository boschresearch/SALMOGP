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
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.likelihoods import SwitchedLikelihood
"""
gpflow.models.GPModel and it's subclasses all have attribute predict_f & likelihood
"""

def argsort_with_same_value_shuffle(x):
    r"""
    basically the same as np.argsort(x),
    but when dealing with duplicated elements in x, the args are randomly ordered
    """
    _, count = np.unique(x, return_counts=True)
    ccs = np.cumsum(count)
    arg_sort = np.argsort(x)
    
    if ccs[-1] == len(count):
        return arg_sort
    
    np.random.shuffle(arg_sort[:ccs[0]])
    for i in range(1, len(count)):
        np.random.shuffle(arg_sort[ccs[i-1]:ccs[i]])
    
    return arg_sort

#class uncertainty_measure(task_indicator):
class uncertainty_measure():
    def full_task_cov(
            self,
            Xnew,
            input_dim:int=1,
            num_task: int=2
            ):
        
        if isinstance(self.likelihood, SwitchedLikelihood):
            """
            in this case the data need to have an additional columns of output task indices
            """
            num_points = np.shape(Xnew)[0]
            
            data = self.add_full_task_idx(Xnew, input_dim, num_task, mode=2)# see task_indicator for details
            _, cov = self.predict_f(data, full_cov=True)
            """
            now we have covariance of all data points,
            but we only want covariance of same X for different tasks,
            which are matrices sitting diagonally in cov
            a.k.a. cov = [cov[..., i:i+num_task, i:i+num_task] for i in np.arange(num_points*num_task)[::num_task]]
            but we want the code to run faster
            """
            cov = tf.transpose(tf.linalg.diag_part(tf.transpose(
                tf.squeeze(
                    tf.split(tf.squeeze(
                        tf.split(cov, num_points, axis=-2)
                        ), num_points, axis=-1)
                    )
                )))
        else:
            _, cov = self.predict_f(Xnew, full_output_cov = True)
        return cov
    
    def multi_output_variance(self, Xnew, input_dim:int=1, num_task: int=2):
        if isinstance(self.likelihood, SwitchedLikelihood):
            num_points = np.shape(Xnew)[0]
            
            data = self.add_full_task_idx(Xnew, input_dim, num_task, mode=1)# see task_indicator for details
            _, var = self.predict_f(data)
            
            var = tf.reshape(var, [num_points, -1])
        else:
            _, var = self.predict_f(Xnew)
        return var
    
    def determinant(
            self, mat
    ) -> tf.Tensor:
        
        return tf.linalg.det(mat)
        
    def entropy(
            self, Xnew, full_output_cov: bool=True, input_dim:int=1, num_task: int=2, partial_output=False
    ) -> tf.Tensor:
        r"""
        compute entropy of (Xnew | X_observed)
        """
        
        if full_output_cov:
            cov = self.full_task_cov(Xnew, input_dim, num_task).numpy()
            # this attribute is from the covariance caculator (see covariance_calculator.py)
            # inheritate the covariance calculator as well
            entropy = 0.5 * tf.math.log(
                    (2*np.pi*np.e)**cov.shape[-1] * \
                    self.determinant(cov)
                    )
            entropy = tf.reshape(entropy, [-1,1])
        else:
            if partial_output:
                _, var = self.predict_f(Xnew)
            else:
                var = self.multi_output_variance(Xnew, input_dim, num_task)
            
            entropy = 0.5 * tf.math.log(
                    (2*np.pi*np.e) * var
                    )
        
        return entropy



class acquisitioner(uncertainty_measure):
    def query_points(
            self, Dpool,
            Xdata=None,
            full_output_cov: bool=True,
            full_task_query: bool=True,
            num_return: int = 1,
            acquition_function: str='entropy',
            return_index: bool=False,
            original_args=None,
            pool_with_task_indicator:bool=False
    ):
        if Dpool[0].shape[0] <= 0:
            raise ValueError("query fail: data pool is empty")
        if full_task_query:
            return self.query_points_full_task(
                Dpool, Xdata=Xdata,
                full_output_cov=full_output_cov,
                num_return=num_return,
                acquition_function=acquition_function,
                return_index=return_index,
                original_args=original_args
                )
        else:
            return self.query_points_partial_task(
                Dpool, Xdata=Xdata,
                full_output_cov=full_output_cov,
                num_return=num_return,
                acquition_function=acquition_function,
                return_index=return_index,
                original_args=original_args
                )
    
    def query_points_full_task(
            self, Dpool,
            Xdata=None,
            full_output_cov: bool=True,
            num_return: int = 1,
            acquition_function: str='entropy',
            return_index: bool=False,
            original_args=None
    ):
        Xpool, Ypool = Dpool
        pool_N, input_dims = np.shape(Xpool)
        _, output_dims = np.shape(Ypool)
        
        if acquition_function.lower() == 'entropy':
            entropy = self.entropy(Xpool, full_output_cov = full_output_cov).numpy().sum(axis=1)
            arg = argsort_with_same_value_shuffle(entropy)[-num_return:]
            
        elif acquition_function.lower() == 'random':
            arg = np.random.randint(0, pool_N, size=num_return)
            
        elif acquition_function.lower() == 'determinant':
            raise ValueError("unfinished acquition function")
            """
            _, cov = self.predict_f(Xpool, full_output_cov=full_output_cov, full_cov=True)
            if full_output_cov:
                cov = tf.constant(np.transpose(cov, (1, 3, 0, 2)))
            """
            
        elif acquition_function.lower() == 'mutual_information':
            raise ValueError("unfinished acquition function")
            """
            return: x = argmax_x MI({y_A, x}, f), where A is the previous training observations
            
            see N. Srinivas, A. Krause, S. M. Kakade, and M. W. Seeger.
            Information-Theoretic Regret Bounds for Gaussian Process Optimization
            in the Bandit Setting. Transactions on Information Theory, 2012
            section II.B
            """
            if Xdata is None:
                raise ValueError('need used training data for this acquisition function')
            
        else:
            raise ValueError("unknown acquition function")
        
        if not original_args is None:
            return_arg = np.array(original_args)[arg]
        else:
            return_arg = arg
        
        
        Dnew = (
            Xpool[arg].reshape([-1, input_dims]),
            Ypool[arg].reshape([-1, output_dims])
            )
        
        if return_index:
            return Dnew, return_arg
        return Dnew
    
    def query_points_partial_task(
            self, Dpool,
            Xdata=None,
            full_output_cov: bool=True,
            num_return: int = 1,
            acquition_function: str='entropy',
            return_index: bool=False,
            original_args=None
    ):
        r"""
        original_args should be 1-D array/list of idx
        """
        Xpool, Ypool = Dpool
        pool_N, input_dim = np.shape(Xpool)
        _, output_dim = np.shape(Ypool)
        
        args_pool = np.arange(pool_N)
        
        # now our pool are data with an additional column of task_idx
        if acquition_function.lower() == 'entropy':
            # determine the most uncertain point(s)
            Xselected = Xpool
            Yselected = Ypool
            
            args_base_pool = np.tile(np.reshape(args_pool, [-1,1]), [1, output_dim]).reshape(-1)
            
            entropy = self.entropy(
                Xselected,
                full_output_cov=False,
                input_dim=input_dim,
                num_task=output_dim,
                partial_output=False
                ).numpy()
            
            # exclude nan entries in Ypool
            entropy = np.where(
                ~np.isnan(Yselected), entropy, -np.inf * np.ones_like(Yselected)
                )
            arg = argsort_with_same_value_shuffle(entropy.reshape(-1))[-num_return:]
            
            args_base = args_base_pool[arg].reshape(-1)
            args_task = arg % output_dim
            
            return_arg = np.vstack((args_base, args_task))
            
        elif acquition_function.lower() == 'random':
            available_entries = np.reshape(~np.isnan(Ypool), [-1]) # exclude nan entries
            args_base_pool = np.tile(np.reshape(args_pool, [-1,1]), [1, output_dim]).reshape(-1)
            args_task_pool = np.tile(np.arange(output_dim), pool_N)
            
            args_pool = np.vstack((
                args_base_pool[available_entries].reshape(-1),
                args_task_pool[available_entries].reshape(-1)
                ))
            
            arg = np.random.randint(0, args_pool.shape[1], size=num_return)
            
            return_arg = args_pool[:, arg].reshape([2,-1])
            
        else:
            raise ValueError("unknown acquition function")
        
        Xnew = Xpool[return_arg[0,:]].reshape([-1, input_dim])
        Ynew = np.nan * np.ones([return_arg.shape[1], output_dim])
        Ynew[np.arange(return_arg.shape[1]), return_arg[1,:]] = Ypool[return_arg[0,:], return_arg[1,:]]
        Dnew = (Xnew, Ynew)
        
        if not original_args is None:
            return_arg[0, :] = np.array(original_args)[return_arg[0, :]] # original_args should be 1-D array/list of idx
        
        if return_index:
            return Dnew, return_arg
        return Dnew



class safety_manager():
    def assign_safety_threshold(
            self,
            Z_threshold: float,
            prob_threshold: float
            ):
        self.Z_threshold = Z_threshold
        self.prob_threshold = prob_threshold
    
    def return_safe_points(
            self,Xpool,
            excluded_args=[],
            Z_threshold: float=None,
            prob_threshold: float=None,
            return_index: bool=False,
            safe_above_threshold: bool=True
    ):
        
        if Z_threshold is not None:
            self.Z_threshold = Z_threshold
        if prob_threshold is not None:
            self.prob_threshold = prob_threshold
        
        excluded_args = np.unique(np.array(excluded_args, dtype=int))
        N, input_dims = np.shape(Xpool)
        used_args = np.delete(np.arange(N, dtype=int), excluded_args)
        X_used_pool = np.reshape(Xpool[used_args], [-1, input_dims])
        
        Z_mu, Z_var = self.predict_f(X_used_pool)
        
        transformed_threshold = (self.Z_threshold - Z_mu.numpy()) / np.sqrt(Z_var)
        pdf = tfp.distributions.Normal(0.0, 1.0)
        cdf = pdf.cdf(transformed_threshold).numpy().reshape(-1)
        ### 
        if safe_above_threshold:
            safe_prob = 1 - cdf
        else:
            safe_prob = cdf
        ###
        arg_safe = safe_prob >= self.prob_threshold
        
        if return_index:
            return X_used_pool[arg_safe].reshape([-1, input_dims]), used_args[arg_safe]
        return X_used_pool[arg_safe].reshape([-1, input_dims])







