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
from typing import Union, Optional
import numpy as np
import tensorflow as tf
import gpflow
import time

from gpflow.utilities import multiple_assign, parameter_dict, read_values
"""
Some pieces of the following classes are adapted from GPflow 2.1.4
(https://github.com/GPflow/GPflow/blob/develop/doc/source/notebooks/advanced/natural_gradients.pct.py
Copyright 2016-2021 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""
class trainer():
    r"""
    inheritate one of the following classes as well, otherwise error may occurs
    gpflow.models.GPModel
    gpflow.models.training_mixins.ExternalDataTrainingLossMixin
    """
    def return_loss(self, data=None):
        if isinstance(self, gpflow.models.training_mixins.ExternalDataTrainingLossMixin):
            if hasattr(self, 'data'):
                data = self.data
            return self.training_loss_closure(data)
        elif isinstance(self,gpflow.models.training_mixins.InternalDataTrainingLossMixin):
            return self.training_loss
    
    def k_DPP(self, k, data):
        raise ValueError('unfinished method')
    
    def training(
            self,
            data=None,
            opt: str='Scipy',
            convergence_check: bool=True,
            **kwargs
            ):
        r"""
        check the following webpage for more info
        https://gpflow.readthedocs.io/en/master/gpflow/optimizers/
        """
        
        loss = self.return_loss(data=data)
        
        if opt.lower() == 'scipy':
            
            timer = -time.perf_counter()
            opt_object = gpflow.optimizers.Scipy()
            optimization_success = False
            while not optimization_success:
                try:
                    opt_res = opt_object.minimize(
                        loss,
                        variables = self.trainable_variables, **kwargs
                        )
                    optimization_success = opt_res.success or not convergence_check
                except:
                    print("Error in optimization - try again")
                    self.model_perturb(0.2)
                if not optimization_success:
                    print("Not converged - try again")
                    self.model_perturb(0.2)    
            
            # if not opt_result.success: re-initialize & re-train
            timer += time.perf_counter()
        
        elif opt.lower() == 'natgrad_adam':
            
            MAXITER = kwargs.pop('MAXITER')
            natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=kwargs.pop('gamma'))
            adam_opt = tf.optimizers.Adam(**kwargs)
            
            timer = -time.perf_counter()
            for _ in range(MAXITER):
                gpflow.set_trainable(self.q_mu, False)
                gpflow.set_trainable(self.q_sqrt, False)
                
                adam_opt.minimize(
                    loss,
                    var_list = self.trainable_variables
                    )
                
                gpflow.set_trainable(self.q_mu, True)
                gpflow.set_trainable(self.q_sqrt, True)
                
                natgrad_opt.minimize(
                    loss,
                    var_list = [(self.q_mu, self.q_sqrt)]
                    )
                
            timer += time.perf_counter()
        
        elif opt.lower() == 'adam':
            MAXITER = kwargs.pop('MAXITER')
            optimizer = tf.optimizers.Adam(**kwargs)
            
            timer = -time.perf_counter()
            for _ in range(MAXITER):
                optimizer.minimize(
                    loss,
                    self.trainable_variables
                    )
            timer += time.perf_counter()
        
        elif opt.lower() == 'kdpp_lbfgsb':
            if hasattr(self, 'inducing_variable') and self.inducing_variable.Z.trainable:
                raise ValueError('this optimization method is only for sparse GPs')
            
            raise ValueError('unfinished method')
        else:
            raise ValueError("unknown optimization method")
        
        return timer
    
    def model_initialization(self, exclude_variables: Union[list, np.ndarray]=[]):
        variables = read_values(self)
        
        for key in variables.keys():
            if key in exclude_variables:
                continue
            variables[key] = np.random.rand() if not hasattr(variables[key], "__len__") \
                else np.random.rand(*np.shape(variables[key]))
                
        multiple_assign(self, variables)
        
        return True
    
    def model_perturb(self, factor_bound:float=0.2):
        for variable in self.trainable_variables:
            unconstrained_value = variable.numpy()
            factor = 1+np.random.uniform(-1*factor_bound,factor_bound,size=unconstrained_value.shape)
            if np.isclose(unconstrained_value,0.0,rtol=1e-07, atol=1e-09).all():
                new_unconstrained_value = (unconstrained_value+np.random.normal(0,0.05,size=unconstrained_value.shape))*factor
            else:
                new_unconstrained_value = unconstrained_value*factor
            variable.assign(new_unconstrained_value)
    
    def best_training(
            self,
            data=None,
            repetition: int=1,
            exclude_variables: Union[list, np.ndarray]=[],
            opt: str='Scipy',
            **kwargs
            ):
        r"""
        the model will be trained with {data} for {repetition} times where
        the model is randomly initialized for each training
        variables in {exclude_variables} will not be reset (initialized)
        
        result with the smallest objective will be kept
        """
        
        if isinstance(self, gpflow.models.training_mixins.ExternalDataTrainingLossMixin):
            best_obj = self.training_loss(data).numpy()
        elif isinstance(self,gpflow.models.training_mixins.InternalDataTrainingLossMixin):
            best_obj = self.training_loss().numpy()
        
        best_pars = read_values(self)
        base_pars = read_values(self)
        timer = 0.0
        
        for i in range(repetition):
            multiple_assign(self, base_pars)
            if not self.model_initialization(exclude_variables):
                raise ValueError(f"initialization failed ({i}th training failed)")
            timer += self.training(data, opt, **kwargs)
            
            if isinstance(self, gpflow.models.training_mixins.ExternalDataTrainingLossMixin):
                new_obj = self.training_loss(data).numpy()
            elif isinstance(self,gpflow.models.training_mixins.InternalDataTrainingLossMixin):
                new_obj = self.training_loss().numpy()
                
            if new_obj < best_obj:
                best_obj = new_obj.copy()
                best_pars = read_values(self)
        
        multiple_assign(self, best_pars)
        return timer



class parameter_manager():
    r"""
    inheritate one of the following classes as well, otherwise error may occurs
    gpflow.models.GPModel
    and it's subclasses
    """
    def get_values_from_model(self, trained_m, target_module=None):
        if target_module is None:
            multiple_assign(
                self,
                parameter_dict(trained_m)
                )
        else:
            multiple_assign(
                target_module,
                parameter_dict(trained_m)
                )
    
    def get_values_from_dict(self, par_dict, target_module=None):
        if target_module is None:
            multiple_assign(self, par_dict)
        else:
            multiple_assign(target_module, par_dict)





