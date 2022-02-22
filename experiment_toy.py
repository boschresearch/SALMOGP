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
import pandas as pd
import os
import sys
import pickle
import datetime
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import functools
print = functools.partial(print, flush=True)
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))

from default_parameters import default_parameters_toy, create_optimizer_args
from my_ALGP_models import ALMOGPR, SafetyMOGPR
from my_plot import *
from my_toolkit import str2bool, raw_wise_compare, data_selection,\
    real_function, safety_function_noise_free,\
    noise_function, save_model, savefile
from my_data_factory import data_manager_toy
from my_model_factory import model_manager

from gpflow.ci_utils import ci_niter
from gpflow.utilities import deepcopy, print_summary

gpflow.config.set_default_summary_fmt("notebook")
tz = datetime.datetime.now().astimezone().tzinfo

def infer(model, safety_model, BI, best_training=False, **optimizer_args):
    if BI:
        model.set_training_args(**optimizer_args)
        timer = model.HMC_sampling()
        safety_model.set_training_args(**optimizer_args)
        timer_s = safety_model.HMC_sampling()
    elif best_training:
        timer = model.best_training(repetition=10, **optimizer_args)
        timer_s = safety_model.best_training(repetition=5, **optimizer_args)
    else:
        timer = model.training(**optimizer_args)
        timer_s = safety_model.training(**optimizer_args)
    
    model.history['training_time'].append(timer)
    safety_model.history['training_time'].append(timer_s)
    
    return True

#np.random.seed(123)
def experiment_pipeline(i_epoch, iter_num_AL, data_manager, POO
                        , model_names, model_configs, optimizer_args, query_num, Z_threshold, prob_threshold
                        , display_figs, save_figs, output_dir):
#######################################################################
########################## Generate datasets ##########################
########################## Generate datasets ##########################
########################## Generate datasets ##########################
########################## Generate datasets ##########################
#######################################################################
    data_manager.set_seed(2000 + i_epoch)
    data_manager.create_tuples()
    X, U, Y, Z = data_manager.create_sample(interval=data_manager.init_interval, num=data_manager.num_init_data)
    _, D, P, T = data_manager.return_dimensions()
    
    X_eval, U_eval, Y_eval, Z_eval = data_manager.true_safe_training_tuple(Z_threshold)
    # create place holders for experimental result

    pred_ref = {'X': X_eval, 'True_Y': Y_eval}
    pred_draw = dict()
    pred_mean = dict()
    RMSE_draw = dict()
    RMSE_mean = dict()
    test_log_density = dict()
    safety_summary = { name: pd.DataFrame(np.zeros([0,4], dtype=int),
                                          columns=['num_all_safe_return', 'num_really_safe', 'query_z', 'query_safe_bool']
                                        ) for name in model_names
    }
    for name in model_names:
        pred_draw[name] = np.zeros([iter_num_AL, *Y_eval.shape])
        pred_mean[name] = np.zeros([iter_num_AL, *Y_eval.shape])
        
        RMSE_draw[name] = np.zeros([iter_num_AL, P])
        RMSE_mean[name] = np.zeros([iter_num_AL, P])
        test_log_density[name] = np.zeros([iter_num_AL, P])
    
#######################################################################
########################### Build our model ###########################
########################### Build our model ###########################
########################### Build our model ###########################
########################### Build our model ###########################
#######################################################################

    mm = model_manager(
        model_names,
        BI=model_configs['BI'], ML=model_configs['ML'], M=model_configs['M']
        )
    m = mm.create_main_models(
        data = (U, Y), MO = model_configs['MO'],
        input_dim = D, y_dim = P,
        training_data_args = None,#np.zeros([2, data_manager.num_init_data], dtype=int),
        N_init = data_manager.num_init_data
    )
    m_safety = mm.create_safety_models(
        data = (U, Z), training_data_args = None,#np.zeros([2, data_manager.num_init_data], dtype=int),
        N_init = data_manager.num_init_data,
        safety_threshold = Z_threshold, safety_prob_threshold = prob_threshold,
        input_dim = D, z_dim=T
    )
    m_safety[model_names[3]].assign_safety_threshold(Z_threshold, -1) # this is a baseline without safety constraint
    """
    I noticed that the very first training always need more than actual time
    """
    warmup_m = ALMOGPR(
        (U,Y),
        gpflow.kernels.LinearCoregionalization([gpflow.kernels.Matern52() for _ in range(P)], W=np.eye(P)),
        np.array([1.0 for _ in range(P)]),
        num_latent_gps = P,
        history_initialize = True
        )
    _ = warmup_m.training(**optimizer_args)
    del warmup_m

    
#######################################################################
########################### ACTIVE LEARNING ###########################
########################### ACTIVE LEARNING ###########################
########################### ACTIVE LEARNING ###########################
########################### ACTIVE LEARNING ###########################
#######################################################################
    for i in range(iter_num_AL):
        if not display_figs:
            plt.clf()
    ##### training
        for name in model_names:
            infer(m[name], m_safety[name], BI=model_configs['BI'], best_training=(i<=-1), **optimizer_args)

    ##### caculate the RMSE for all models
        for name in model_names:
        # draw based
            pred_draw[name][i, :, :] = m[name].predict_f_samples(U_eval, full_cov=False, full_output_cov=False).numpy()
            RMSE_draw[name][i, :] = np.sqrt(np.mean(
                        np.power(Y_eval - pred_draw[name][i, :, :], 2),
                        axis=0
                        ))
        # mean based
            mu, _ = m[name].predict_f(U_eval)
            pred_mean[name][i, :, :] = mu.numpy()
            RMSE_mean[name][i, :] = np.sqrt(np.mean(
                        np.power(Y_eval - mu.numpy(), 2),
                        axis=0
                        ))

    ##### caculate the log_density of (U_eval, Y_eval) for all models
            test_log_density[name][i, :] = m[name].predict_log_density_full_output((U_eval, Y_eval)).numpy().sum(axis=0)
        
    ##### dertermine safe points
        D_safe = {}
        _, U_explor, _, _ = data_manager.training_pool
        for name in model_names:
            _, args_safe = m_safety[name].return_safe_points(U_explor, return_index=True)
            D_safe[name] = data_selection(data_manager.training_pool, args_safe)
            
        
    ##### plot & save models
        if save_figs or display_figs:
            for name in model_names:
                plot_model_series(
                        m[name], data_manager,
                        [min(D_safe[name][0][:,0]), max(D_safe[name][0][:,0])],
                        [save_figs, os.path.join(output_dir, "models", f"exp_{i_epoch}_data_std{data_noise_std[0]}&{data_noise_std[1]}_{name}_iter{i}.jpg")],
                        [-2, 3], display_fig=display_figs
                        )

                plot_safety_label_series(
                    m_safety[name], data_manager, safety_X = [min(D_safe[name][0][:,0]), max(D_safe[name][0][:,0])],
                    savefile=[save_figs, os.path.join(output_dir, "models", f"exp_{i_epoch}_data_std{data_noise_std[0]}&{data_noise_std[1]}_{name}_safety_model_iter{i}.jpg")],
                    display_fig=display_figs
                    )

                save_model(os.path.join(output_dir, "models", f"exp_{i_epoch}_data_std{data_noise_std[0]}&{data_noise_std[1]}_{name}_iter{i}.txt"), m[name])

    ##### print training info
        if i==iter_num_AL-1:
            print("##############################################"+\
                  "\n######                                  ######"+\
                  "\n######       experiment %7d         ######"%(i_epoch)+\
                  "\n######       iteration  %4d/%4d       ######"%(i+1, iter_num_AL)+\
                  "\n######                                  ######"+\
                  "\n######             Training             ######")
            for ind in range(len(model_names)):
                print("###### %-20s: %7.3f(s) ######"%(model_names[ind], m[model_names[ind]].history["training_time"][-1]))
                print("######         safety_model: %7.3f(s) ######"%(m_safety[model_names[ind]].history["training_time"][-1]))
        
            print("######                                  ######"+\
                  "\n######         Done Training            ######"+\
                  "\n######                                  ######"+\
                  "\n##############################################\n")
            break
        
    ##### update datasets for next iteration
        # determine the most uncertain point(s) among safe points
        for j, name in enumerate(model_names):
            _, U_pool, _, Z_safe = D_safe[name]
            unused_idx = ~raw_wise_compare(U_pool, m[name].data[0])
            D_pool = data_selection(D_safe[name], unused_idx)

            if j == 1:
                _, args_new = m[name].query_points(
                        D_pool[1:3],
                        num_return = query_num,
                        acquition_function='random',
                        full_task_query=not POO,
                        return_index=True
                        )
            
            else:
                _, args_new = m[name].query_points(
                        D_pool[1:3],
                        full_output_cov = True,
                        num_return = query_num,
                        acquition_function = 'entropy',
                        full_task_query = not POO,
                        return_index=True
                        )
            
            _, U_new, Y_new, Z_new = data_selection(D_pool, args_new)
            
            m[name].update_dataset((U_new, Y_new))
            m_safety[name].update_dataset((U_new, Z_new))

            safety_summary[name] = safety_summary[name].append(
                pd.DataFrame([[
                    Z_safe.shape[0],  # the total points returned by model as safe
                    np.sum(Z_safe > m_safety[name].Z_threshold),  # how many are actually safe
                    Z_new,
                    Z_new > m_safety[name].Z_threshold
                ]], index=pd.RangeIndex(i, i+1), columns=safety_summary[name].columns.values
                )
            )
            
        print("##############################################"+\
              "\n######                                  ######"+\
              "\n######       experiment %7d         ######"%(i_epoch)+\
              "\n######       iteration  %4d/%4d       ######"%(i+1, iter_num_AL)+\
              "\n######                                  ######"+\
              "\n######             Training             ######")
        for ind in range(len(model_names)):
            print("###### %-20s: %7.3f(s) ######"%(model_names[ind], m[model_names[ind]].history["training_time"][-1]))
            print("######         safety_model: %7.3f(s) ######"%(m_safety[model_names[ind]].history["training_time"][-1]))
        
        print("######                                  ######"+\
              "\n######         Datapoints query         ######")
        for ind in range(len(model_names)):
            print("###### %-20s: %7.3f(s) ######"%(model_names[ind], m[model_names[ind]].history["point_selection_time"][-1]))
        
        print("######                                  ######"+\
              "\n##############################################\n")

    return m, m_safety, pred_ref, pred_draw, pred_mean, RMSE_draw, RMSE_mean, test_log_density, safety_summary





if __name__ == "__main__":
    pars = default_parameters_toy()
    parser = argparse.ArgumentParser(description='Run ALSVGPvsOthers, toy dataset.')
    # experiment settings
    parser.add_argument('--POO', default= pars.POO, type=str2bool, nargs='?', const=True, help=f"defauls={pars.POO}, whether the outputs are partially observed or not")
    parser.add_argument('--experiment_index', default= pars.experiment_index, type=int, help=f"defauls={pars.experiment_index}, if we run this script in parallel, we would need indices to distinguish different trials")
    parser.add_argument('--repetition', default= pars.repetition, type=int, help=f"default={pars.repetition}, number of run experiments, notice that the experiments are NOT run in parallel")
    parser.add_argument('--iteration_num', default= pars.iteration_num, type=int, help=f"default={pars.iteration_num}, number of active learning iteration")
    # experiment settings, data & models & functions
    parser.add_argument('--num_init_data', default=pars.num_init_data, type=int, help=f"default={pars.num_init_data}, number initial data points")
    parser.add_argument('--series_half_dim', default=pars.series_half_dim, type=int, help=f"default={pars.series_half_dim}, time series half_dim, model input has 2*half_dim + 1 columns being [..., x-step*1, x, x+step*1, ...]")
    parser.add_argument('--series_step', default=pars.series_step, type=float, help=f"default={pars.series_step}, time series step, model input = [..., x-step*1, x, x+step*1, ...]")
    parser.add_argument('--fullGP', default=pars.fullGP, type=str2bool, nargs='?', const=True, help=f"default={pars.fullGP}, whether we want full GP or (S)VGP, when True, M is useless")
    parser.add_argument('--bayesian', default=pars.bayesian, type=str2bool, nargs='?', const=True, help=f"default={pars.bayesian}, whether we run Bayesian inference or not, only for full GP")
    parser.add_argument('--M', default= pars.M, type=int, help=f"default={pars.M}, number of inducing points")
    parser.add_argument('--query_num', default= pars.query_num, type=int, help=f"default={pars.query_num}, number of query points in each active learning iteration")
    parser.add_argument('--optimizer', default=pars.optimizer, type=str, help=f"default={pars.optimizer}, optimizer (currently 'scipy', 'natgrad_adam', or 'adam', case-insensitive)")
    parser.add_argument('--exploration_interval', default= pars.exploration_interval, nargs=2, type=float, help=f"default={pars.exploration_interval}, X interval for active learning exploration, notice that the RMSE and log_density are also computed with samples in this interval")
    parser.add_argument('--initial_interval', default= pars.initial_interval, nargs=2, type=float, help=f"default={pars.initial_interval}, X interval for drawing initial training sets")
    parser.add_argument('--data_noise_std', default= pars.data_noise_std, nargs=2, type=float, help=f"default={pars.data_noise_std}, standard deviation of observations, 2 output dims")
    parser.add_argument('--data_noise_std_safety', default= pars.data_noise_std_safety, type=float, help=f"default={pars.data_noise_std_safety}, standard deviation of safety label observations")
    # experiment settings, safety constraint
    parser.add_argument('--safety_threshold', default= pars.safety_threshold, type=float, help=f"default={pars.safety_threshold}, safe when safety label is above this threshold")
    parser.add_argument('--safety_prob_threshold', default= pars.safety_prob_threshold, type=float, help=f"default={pars.safety_prob_threshold}, safe when p(safety label above safety_threshold) >= this prob_threshold")
    # save result
    parser.add_argument('--display_figs', default= pars.display_figs, type=str2bool, nargs='?', const=True, help=f"default={pars.display_figs}, whether the figures are shown")
    parser.add_argument('--save_figs', default= pars.save_figs, type=str2bool, nargs='?', const=True, help=f"default={pars.save_figs}, whether the figures are saved")
    parser.add_argument('--output_dir', default= pars.output_dir, type=str, help=f"default={pars.output_dir}, where to save the results")
    
    args = parser.parse_args()

    optimizer_args = create_optimizer_args(args.optimizer)
    
    exp_idx = args.experiment_index
    iter_num_AL = args.iteration_num
    Z_threshold = args.safety_threshold
    prob_threshold = args.safety_prob_threshold
    exp_interval = args.exploration_interval

    data_noise_std = np.array(args.data_noise_std)
    data_noise_std_safety = args.data_noise_std_safety

    num_init_data = args.num_init_data
    query_num = args.query_num

    model_names = ["AL_MOGP", "RS_MOGP", "AL_indGPs", "AL_MOGP_nosafe"]
    MO = [True, True, False, True]
    ML = args.fullGP # if False, then variational inference
    BI = args.bayesian and ML
    M = args.M
    model_configs = {'MO':MO, 'ML':ML, 'BI':BI, 'M':M}

    output_dir = args.output_dir

    if BI:
        folder_key = 'BGP'
    elif ML:
        folder_key = 'fullGP'
    else:
        folder_key = f'M{M}'
    
    if args.POO:
        output_dir += f"_POO"
        output_dir = os.path.join(output_dir, folder_key)
    else:
        output_dir = os.path.join(output_dir, folder_key)
    
    Path(os.path.join(output_dir, "pkl_files", "individual_trials")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, "models")).mkdir(parents=True, exist_ok=True)

    true_func = real_function
    true_safe_func = safety_function_noise_free
    exp_data_manager = data_manager_toy(
        exp_interval,
        true_function=true_func,
        true_safety_function=true_safe_func,
        noise_function = noise_function,
        noise_std=data_noise_std,
        noise_std_safety=data_noise_std_safety,
        num_init_data=num_init_data,
        input_dim= 1,#args.input_dim,
        init_interval=args.initial_interval,
        series_half_step_num=args.series_half_dim,
        series_step=args.series_step,
        POO=args.POO
    )

    def run(i):
        
        with open(os.path.join(output_dir, f'data_std{data_noise_std[0]}&{data_noise_std[1]}_experiment_setup.txt'), 'w') as fp:
            time_now = datetime.datetime.now(tz = tz)
            print(time_now.strftime('%Z (UTC%z)\n%Y.%b.%d  %A  %H:%M:%S\n'), file = fp)
            print(f"input dimension                    : {exp_data_manager.input_dim}", file=fp)
            print(f"data noise / safety noise (std)    : {data_noise_std} / {data_noise_std_safety}", file=fp)
            print("AL, num of initial training points : %d"%(num_init_data), file = fp)
            print(f"AL, initial interval               : {exp_data_manager.init_interval}", file=fp)
            print("AL, num of iterations              : %d"%(iter_num_AL), file = fp)
            print(f"AL, exploration interval           : {exp_interval}", file = fp)
            print("AL, num of quering points          : %d"%(query_num), file = fp)
            print("safety threshold (probability)     : >%f (with p>=%.3f)"%(Z_threshold, prob_threshold), file = fp)
        
        
        m, m_safety, pred_ref, pred_draw, pred_mean, RMSE_draw, RMSE_mean, test_log_density, safety_summary = \
        experiment_pipeline(i, iter_num_AL, exp_data_manager, args.POO
                        , model_names, model_configs, optimizer_args, query_num, Z_threshold, prob_threshold
                        , args.display_figs, args.save_figs, output_dir)
        
        Training_time = {
            model_names[ind]: m[model_names[ind]].history["training_time"] for ind in range(len(model_names))
            }
        data_selection_time = {
            model_names[ind]: m[model_names[ind]].history["point_selection_time"] for ind in range(len(model_names))
            }
        
        savefile(os.path.join(output_dir, "pkl_files", "pred_ref.pkl"),
                 pred_ref,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_draw_exp{i}.pkl"),
                 pred_draw,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_mean_exp{i}.pkl"),
                 pred_mean,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw_exp{i}.pkl"),
                 RMSE_draw,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean_exp{i}.pkl"),
                 RMSE_mean,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_test_log_density_exp{i}.pkl"),
                 test_log_density,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_Training_time_exp{i}.pkl"),
                 Training_time,
                 mode = "wb")
        savefile(os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_data_selection_time_exp{i}.pkl"),
                 data_selection_time,
                 mode = "wb")

        fullpath_noextention = os.path.join(output_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_safety_summary_exp{i}")
        savefile(fullpath_noextention+'.pkl', safety_summary, mode='wb')
        with open(fullpath_noextention+'.txt', mode='w') as fp:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.width', 2000)

            for name in model_names:
                print(f'model: {name}', file=fp)
                print(f'{safety_summary[name]}\n\n', file=fp)

    for i in range(args.repetition):
        run(exp_idx*1000 + i)
    