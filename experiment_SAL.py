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
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from gpflow.utilities import deepcopy, print_summary
from default_parameters import default_parameters_OWEO, create_optimizer_args
from my_toolkit import str2bool, loadfile
from my_experiments import experiment_manager
from my_data_factory import data_manager_OWEO
from my_model_factory import model_manager

gpflow.config.set_default_summary_fmt("notebook")
gpus = tf.config.experimental.list_physical_devices('GPU')

tz = datetime.datetime.now().astimezone().tzinfo
"""
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
"""
if __name__ == "__main__":
    pars = default_parameters_OWEO()
    parser = argparse.ArgumentParser(description='Run ALSVGPvsOthers on OWEO dataset.')
    # experiment settings
    parser.add_argument('--experiment_index',       default=pars.experiment_index, type=int, help=f"default={pars.experiment_index}, if we run this script in parallel, we would need indices to distinguish different trials")
    parser.add_argument('--repetition',             default=pars.repetition, type=int, help=f"default={pars.repetition}, number of experiments run in this script, notice that the experiments are NOT run in parallel")
    parser.add_argument('--input_dir',              default=pars.raw_data_dir, type=str, help=f"default={pars.raw_data_dir}, where the input files are")
    parser.add_argument('--filename_training',      default=pars.filename_training, type=str, help=f"default={pars.filename_training}, full path of training data")
    parser.add_argument('--filename_test',          default=pars.filename_test, type=str, help=f"default={pars.filename_test}, full path test data")
    parser.add_argument('--used_y_ind',             default=pars.used_y_ind, nargs='+', type=int, help=f"default={pars.used_y_ind}, use subsets of [0, 1, 2, 3, 4, 5]")
    parser.add_argument('--used_z_ind',             default=pars.used_z_ind, type=int, help=f"default={pars.used_z_ind}, use 6 or 7")
    parser.add_argument('--iteration_num',          default=pars.iteration_num, type=int, help=f"default={pars.iteration_num}, number of active learning iteration")
    # experiment settings, data & models & functions
    parser.add_argument('--optimizer',              default=pars.optimizer, type=str, help=f"default={pars.optimizer}, optimizer (currently 'scipy', 'natgrad_adam', or 'adam', case-insensitive)")
    parser.add_argument('--preselect_data_num',     default=None, type=int, help=f"default={None}, if we want to run AL only on a subset of training data")
    parser.add_argument('--num_init_data',          default=pars.num_init_data, type=int, help=f"default={pars.num_init_data}, number initial data points")
    parser.add_argument('--fullGP',                 default=pars.fullGP, type=str2bool, nargs='?', const=not pars.fullGP, help=f"default={pars.fullGP}, whether we want full GP or (S)VGP, when True, M is useless")
    parser.add_argument('--bayesian',               default=pars.bayesian, type=str2bool, nargs='?', const=not pars.bayesian, help=f"default={pars.bayesian}, whether we run Bayesian inference or not, only for full GP")
    parser.add_argument('--M',                      default=pars.M, type=int, help=f"default={pars.M}, number of inducing points")
    parser.add_argument('--query_num',              default=pars.query_num, type=int, help=f"default={pars.query_num}, number of query points in each active learning iteration")
    parser.add_argument('--fixed_initial_dataset',  default=pars.fixed_initial_dataset, type=str2bool, nargs='?', const=not pars.fixed_initial_dataset, help=f"default={pars.fixed_initial_dataset}, whether we want to fix the initial training set for all experiment trials")
    parser.add_argument('--fixed_models',           default=pars.fixed_models, type=str2bool, nargs='?', const=not pars.fixed_models, help=f"default={pars.fixed_models}, whether we want to fix the model parameters (need model parameter files)")
    parser.add_argument('--share_kernel',           default=pars.share_kernel, type=str2bool, nargs='?', const=not pars.share_kernel, help=f"default={pars.share_kernel}, whether we want to use the same parameters for stationary kernel across different outputs")
    # experiment settings, safety constraint
    parser.add_argument('--safety_threshold',       default=pars.safety_threshold, type=float, help=f"default={pars.safety_threshold}, safe when safety label is above this threshold")
    parser.add_argument('--safety_prob_threshold',  default=pars.safety_prob_threshold, type=float, help=f"default={pars.safety_prob_threshold}, safe when p(safety label above safety_threshold) >= this prob_threshold")
    # save result
    parser.add_argument('--display_figs',           default=pars.display_figs, type=str2bool, nargs='?', const=not pars.display_figs, help=f"default={pars.display_figs}, whether the figures are shown")
    parser.add_argument('--save_figs',              default=pars.save_figs, type=str2bool, nargs='?', const=not pars.save_figs, help=f"default={pars.save_figs}, whether the figures are saved")
    parser.add_argument('--save_kernel_figs',       default=pars.save_kernel_figs, type=str2bool, nargs='?', const=not pars.save_kernel_figs, help=f"default={pars.save_kernel_figs}, whether the kernel plots are saved")
    parser.add_argument('--save_models',            default=pars.save_models, type=str2bool, nargs='?', const=not pars.save_models, help=f"default={pars.save_models}, whether the models' parameters are saved")
    parser.add_argument('--save_step',              default=pars.save_step, type=int, help=f"default={pars.save_step}, plot/save the models once every {pars.save_step} iteration(s)")
    parser.add_argument('--output_dir',             default=pars.output_dir, type=str, help=f"default={pars.output_dir}, where to save the results")
    args = parser.parse_args()
    
    exp_idx = args.experiment_index
    
    BI = args.fullGP and args.bayesian
    
    if BI:
        output_dir = os.path.join(args.output_dir, "BGP")
    elif args.fullGP:
        output_dir = os.path.join(args.output_dir, "fullGP")
    else:
        output_dir = os.path.join(args.output_dir, f"M{args.M}")
    
    num_init_data = int( args.num_init_data / len(args.used_y_ind) )
    num_AL_iter = args.iteration_num
    
    if num_AL_iter > 0 and args.preselect_data_num is None:
        output_dir = os.path.join(output_dir, f"AL_n0_{num_init_data}")
    elif num_AL_iter > 0 and not args.preselect_data_num is None:
        output_dir = os.path.join(output_dir, f"AL_onSubset_n0_{num_init_data}")
    elif num_AL_iter == 0:
        output_dir = os.path.join(output_dir, "model_test")
    else:
        raise ValueError('num of AL iteration must be 0 (test model) or a positive integer')
    
    data_training = loadfile(args.filename_training, mode='rb')
    data_test = loadfile(args.filename_test, mode='rb')
    
    ### the name of models we used in experiments,
    if num_AL_iter == 0:
        model_names = ['MOGP', 'indGPs']
        MO = np.array([True, False], dtype=bool)
        query_policy = {}
    else:
        model_names = ["AL_MOGP", "RS_MOGP", "AL_indGPs", "AL_MOGP_nosafe"]
        MO = np.array([True, True, False, True], dtype=bool)
        query_policy = { # [full_output_cov: bool, acquisition_function: str]
                model_names[0]: {'full_output_cov':True,'acquition_function':'entropy'},
                model_names[1]: {'full_output_cov':False,'acquition_function':'random'},
                model_names[2]: {'full_output_cov':True,'acquition_function':'entropy'},
                #model_names[3]: {'full_output_cov':False,'acquition_function':'entropy'}
                model_names[3]: {'full_output_cov':True,'acquition_function':'entropy'}
            }
    
    ##############################################################
    ##################### create data_manager ####################
    ##################### create data_manager ####################
    ##################### create data_manager ####################
    ##################### create data_manager ####################
    ##################### create data_manager ####################
    ##############################################################
    exp_data_manager = data_manager_OWEO(data_training, data_test)
    exp_data_manager.create_tuples(args.used_y_ind, args.used_z_ind)
    
    args_idx = 1 if args.fixed_initial_dataset else exp_idx % 100
    
    training_data_args = exp_data_manager.generate_training_args(
        args.input_dir, num_init_data, args_idx,
        safety_threshold=args.safety_threshold if num_AL_iter > 0 else None,
        safe_above_threshold=False,
        num_preselect=args.preselect_data_num
    )
    
    if args.fullGP or args.M == 0:
        X_init, U_init, Y_init, Z_init = \
            exp_data_manager.training_data_selection(training_data_args)
        N_init = None
    else:
        exp_data_manager.training_data_initializer(model_names, training_data_args)
        N_init = len(training_data_args)
        X_init = U_init = Y_init = Z_init = None
    
    _, u_dim, y_dim, z_dim = exp_data_manager.return_dimensions()
    ##############################################################
    ###################### create our models #####################
    ###################### create our models #####################
    ###################### create our models #####################
    ###################### create our models #####################
    ###################### create our models #####################
    ##############################################################
    mm = model_manager(model_names, BI, args.fullGP, args.M)
    m = mm.create_main_models(
        data=(U_init, Y_init),
        MO=MO,
        input_dim=u_dim,
        y_dim=y_dim,
        training_data_args=training_data_args,
        N_init=N_init,
        kernel_share_parameters=args.share_kernel
    )
    m_safety = mm.create_safety_models(
        data=(U_init, Z_init),
        training_data_args=training_data_args,
        N_init=N_init,
        safety_threshold=args.safety_threshold,
        safety_prob_threshold=args.safety_prob_threshold,
        input_dim=u_dim,
        z_dim=z_dim
    )
    if num_AL_iter > 0:
        m_safety[model_names[3]].assign_safety_threshold(args.safety_threshold, -1)
    
    if args.fixed_models:
        
        warmup_m = None
        path = os.path.join(output_dir, "pkl_files")
        lik_all = loadfile(os.path.join(path, "all_likelihood_variances.pkl"))
        path = os.path.join(path, "individual_trials")
        
        #output_dir += '_fixed_model_pars' 
        output_dir += '_fixed_model_pars_fixed_all_exp' 
        
        for name in model_names:
            #kern_par = loadfile(os.path.join(path, f'kernel_{name}_exp{exp_idx}_iter{num_AL_iter-1}.pkl'))
            kern_par = loadfile(os.path.join(path, f'kernel_{name}_exp1_iter{num_AL_iter-1}.pkl'))
            m[name].get_values_from_dict(kern_par, m[name].kernel)
            gpflow.set_trainable(m[name].kernel, False)
            
            lik_par = {'.variance': np.array([lik_all[exp_idx][name+f'{j+1}'][num_AL_iter-1] for j in range(y_dim)])}
            m[name].get_values_from_dict(lik_par, m[name].likelihood)
            gpflow.set_trainable(m[name].likelihood, False)
    else:
        warmup_m = deepcopy(m[model_names[np.squeeze(np.where(~MO))]])
    ##############################################################
    ####################### run experiment #######################
    ####################### run experiment #######################
    ####################### run experiment #######################
    ####################### run experiment #######################
    ####################### run experiment #######################
    ##############################################################
    for i in range(args.repetition):
        exp_pipeline = experiment_manager(
            seed = 2020 + 1000*exp_idx + i,
            exp_idx = exp_idx,
            data_manager=exp_data_manager,
            model_dictionary = m,
            safety_model_dictionary = m_safety,
            optimizer_args = create_optimizer_args(args.optimizer),
            fixed_initial_dataset = args.fixed_initial_dataset,
            iter_num_AL = num_AL_iter,
            query_policy = query_policy,
            query_num = args.query_num,
            output_dir = output_dir,
            save_models = args.save_models,
            save_figs = args.save_figs,
            save_every_N_step = args.save_step,
            display_figs = args.display_figs,
            plot_kernels = args.save_kernel_figs,
            save_iv = False,#(exp_idx == 0 and args.M != 0),
            template_model=warmup_m,
            bayesian_inference = BI
            )
        
        exp_pipeline.run_pipeline()













