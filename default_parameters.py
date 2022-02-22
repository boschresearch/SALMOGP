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
import numpy as np
class default_parameters_toy():
    def __init__(self):
        # experiment settings
        self.POO = True
        self.experiment_index = 0
        self.repetition = 1
        # experiment settings, data & models & functions
        self.num_init_data = 12
        self.iteration_num = 40
        self.query_num = 1
        self.fullGP = True
        self.bayesian = True
        self.M = 10
        self.optimizer = 'adam' # 'scipy', 'natgrad_adam', or 'adam', case-insensitive
        self.initial_interval = [-0.6, 0.6]
        self.exploration_interval = [-2.0, 2.0]
        self.input_dim = 1
        self.series_half_dim = 0 # ignore this
        self.series_step = 0.2 # ignore this
        self.data_noise_std = 0.4 * np.ones(2)#[0.4, 0.1]
        self.data_noise_std_safety = 0.05
        # experiment settings, safety constraint
        self.safety_threshold = 0.7
        self.safety_prob_threshold = 0.95
        # save result
        self.display_figs = False
        self.save_figs = True
        self.output_dir = os.path.join("experimental_result", "toy")

class default_parameters_GPsamples():
    def __init__(self):
        # experiment settings
        self.POO = True
        self.experiment_index = 0
        self.repetition = 1
        # experiment settings, data & models & functions
        self.num_init_data = 40
        self.iteration_num = 40
        self.query_num = 1
        self.fullGP = True
        self.bayesian = False
        self.M = 10
        self.optimizer = 'adam' # 'scipy', 'natgrad_adam', or 'adam', case-insensitive
        self.input_dim = 2
        self.latent_dim = 3 # only used to generate data
        self.output_dim = 4
        self.data_noise_std = 0.4 * np.ones(4)
        self.data_noise_std_safety = 0.05
        # experiment settings, safety constraint
        self.safety_threshold = -0.56
        self.safety_prob_threshold = 0.95
        # save result
        self.display_figs = False
        self.save_figs = False
        self.input_dir = os.path.join("data", "GP_samples")
        self.output_dir = os.path.join("experimental_result", "GPdata", "X2L3Y4")


class default_parameters_OWEO():
    def __init__(self):
        # data preprocessing
        self.used_measurement_training = [10,11,12,13,14,15,16,17,18,19,30,31,32,33,34,35,36,37,38,39]
        self.used_measurement_test = [53,54,55,56,57,58,59,60,61,62,63,64,65]
        #self.NX_pos_str = ['r0c0', 'r1c1', 'r1c2', 'r1c3', 'r2c1', 'r2c2', 'r2c3', 'r3c3', 'r3c4', 'r4c0', 'r4c4', 'r7c1', 'r7c2', 'r12c2'] # for CO2 & HC
        self.NX_pos_str = ['r0c0', 'r1c0', 'r1c1', 'r1c2', 'r1c3', 'r2c1', 'r3c1', 'r3c3', 'r4c0', 'r4c2', 'r4c3', 'r7c1', 'r8c2'] # for HC & O2
        #self.pt1_X = [103, 32, 49, 126, 20] # for CO2 & HC
        self.pt1_X = [298, 38, 18, 32, 28] # for HC & O2
        self.pt1_Y = [1, 1, 1, 1, 1, 1, 1, 1]
        self.raw_data_dir = "data"
        self.filename_training = os.path.join("data", "data_training_HC_O2.pkl")
        self.filename_test = os.path.join("data", "data_test_HC_O2.pkl")
        # experiment settings
        self.experiment_index = 0
        self.repetition = 1
        self.used_y_ind = [3, 5]#[2, 3, 5] # = [CO2, HC, O2]
        self.used_z_ind = 6
        # experiment settings, data & models & functions
        self.fixed_initial_dataset = False
        self.num_init_data = 48
        self.iteration_num = 40
        self.query_num = 1
        self.fullGP = True
        self.bayesian = True
        self.M = 150
        self.fixed_models = False
        self.share_kernel = False
        # optimizer settings
        self.optimizer = 'scipy' # 'scipy', 'natgrad_adam', or 'adam', case-insensitive
        # experiment settings, safety constraint
        self.safety_threshold = 1.0
        self.safety_prob_threshold = 0.95
        # save result
        self.display_figs = False
        self.save_figs = False
        self.save_kernel_figs = False
        self.save_models = True
        self.save_step = 10 # save figs & models every N step
        self.output_dir = os.path.join("experimental_result", "OWEO_HC_O2")




def create_optimizer_args(opt:str='adam'):
    from gpflow.ci_utils import ci_niter
    if opt.lower() == 'scipy':
        return {
            'opt': 'scipy',
            'options':{"maxiter": ci_niter(500)},
            'method': 'L-BFGS-B'
            }
    
    elif opt.lower() == 'adam':
        return {
            'opt': 'adam',
            'MAXITER': ci_niter(500),
            'learning_rate': 1e-1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-07,
            'amsgrad': False
            }
    
    elif opt.lower() == 'natgrad_adam':
        return {
            'opt': 'natgrad_adam',
            'MAXITER': ci_niter(200),
            'gamma': 0.1,
            'learning_rate': 1e-1
            }
    
    else:
        raise ValueError("unknown optimizer")





        