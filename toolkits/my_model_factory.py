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
import gpflow
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from typing import Union, Sequence, Optional
from gpflow.utilities import deepcopy#, print_summary
from my_ALGP_models import ALMOGPR, SafetyMOGPR
from my_Bayesian_GP_models import ALBMOGPR, SafetyBMOGPR
from modules.likelihood import MultiGaussian

class model_manager():
    def __init__(
            self,
            model_names: Sequence[str],
            BI: bool, # when ML == True, decide whether we want to use bayesian inference on hyperpars
            ML: bool, # use marginal likelooh, if False, variational models will be used
            M: int=0 # if ML == False, this determine the number of inducing points (0 means VI but no sparcity)
        ):

        self.names = model_names
        self.BI = ML and BI
        self.ML = ML
        self.M = M


    def create_main_models(
            self,
            data,
            MO: Sequence[bool],
            input_dim: int,
            y_dim: int,
            training_data_args: Union[list, np.ndarray],
            N_init: int,
            kernel_share_parameters: bool=False,
            share_inducing_points: bool=True
    ):
        m = {}
        for j, name in enumerate(self.names):
            if not MO[j]: # independent outputs model
                kern = gpflow.kernels.SeparateIndependent(
                    [gpflow.kernels.Matern52() for _ in range(y_dim)]
                )
            else:
                if kernel_share_parameters:
                    kern = gpflow.kernels.Matern52()
                    kern_list = [kern for _ in range(y_dim)]
                else:
                    kern_list = [gpflow.kernels.Matern52() for _ in range(y_dim)]

                kern = gpflow.kernels.LinearCoregionalization(kern_list, W=np.eye(y_dim))

            if self.ML:
                if not self.BI:
                    model_class = ALMOGPR
                else:
                    model_class = ALBMOGPR

                m[name] = model_class(
                    data, kern,
                    history_initialize=True,
                    data_args=training_data_args
                )

            elif self.M == 0:
                raise ValueError("this implementation only supports max. likelihood or Bayesian treatment")
            else:
                raise ValueError("this implementation only supports max. likelihood or Bayesian treatment")

        return m

    def create_safety_models(
            self,
            data,
            training_data_args: Union[list, np.ndarray],
            N_init:int,
            safety_threshold:float,
            safety_prob_threshold:float,
            input_dim:int,
            z_dim:int=1
    ):
        m_safety = {}
        for ind, name in enumerate(self.names):
            kern = gpflow.kernels.SeparateIndependent([gpflow.kernels.Matern52() for _ in range(z_dim)])
            lik = MultiGaussian()
            if self.ML:
                if not self.BI:
                    model_class = SafetyMOGPR
                else:
                    model_class = SafetyBMOGPR

                m_safety[name] = model_class(
                    data, kern,
                    history_initialize=True, data_args=training_data_args
                )

            elif self.M == 0:
                raise ValueError("this implementation only supports max. likelihood or Bayesian treatment")
            else:
                raise ValueError("this implementation only supports max. likelihood or Bayesian treatment")

            m_safety[name].assign_safety_threshold(safety_threshold, safety_prob_threshold)

        return m_safety
