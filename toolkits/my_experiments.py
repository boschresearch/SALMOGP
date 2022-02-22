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
import pandas as pd
import os
import sys
import time
import argparse
import datetime
from matplotlib import pyplot as plt
import functools
print = functools.partial(print, flush=True)

from pathlib import Path
from gpflow.ci_utils import ci_niter
from gpflow.utilities import deepcopy, print_summary
from default_parameters import default_parameters_OWEO, create_optimizer_args
sys.path.append(os.path.dirname(__file__))
from my_plot import *
from my_ALGP_models import myMOGPR
from my_toolkit import data_selection, args_partially_queried_matrix, extract_name_tuple, extract_data_tuple, result_placeholder, loadfile, savefile, save_model, save_model_dict, create_args
from modules.likelihood import MultiGaussian

gpflow.config.set_default_summary_fmt("notebook")
gpus = tf.config.experimental.list_physical_devices('GPU')

tz = datetime.datetime.now().astimezone().tzinfo


def infer(model, BI, best_training=False, data=None, **optimizer_args):
    if BI:
        model.set_training_args(**optimizer_args)
        timer = model.HMC_sampling()
    elif best_training:
        timer = model.best_training(data=data, repetition=10, **optimizer_args)
    else:
        timer = model.training(data=data, **optimizer_args)
    
    model.history['training_time'].append(timer)
    
    return True



class experiment_manager():
    def __init__(
            self,
            seed: int,
            exp_idx: int,
            data_manager, # the manager should be ready for first training, i.e. datasets initialized
            model_dictionary: dict,
            safety_model_dictionary: dict,
            optimizer_args: dict,
            fixed_initial_dataset: bool,
            iter_num_AL: int,
            query_policy: dict,
            query_num: int,
            output_dir: str,
            save_models: bool,
            save_figs: bool,
            save_every_N_step: int = 10,
            display_figs: bool = False,
            plot_kernels: bool = False,
            save_iv: bool = False,
            template_model = None,
            template_safety_model = None,
            partially_observed_output: bool=False,
            bayesian_inference: bool=False
            ):
        r"""
        data_manager: object,
                should be ready for training, i.e. datasets initialized
                
                no need to tell experiment_manager if the models are data_inclusive or not,
                but data_manager, model_dictionary, safety_model_dictionary should be all ready
        model_dictionary:
                dictionary of main models
        safety_model_dictionary:
                dictionary of safety models, keys should be the same as those of model_dictionary
        optimizer_args:
                dictionary of opt type, MAXITER per training, kwargs to be passed into the optimizers
        fixed_initial_dataset:
                as the data_manager control the data, this pars exists only for documentation
        iter_num_AL:
                number of AL iterations
        query_policy: dictionary, keys are the same as models' keys',
                query_policy[name] are all [full_output_cov: bool, acquisition_function: str (case insensitive)]
        query_num:
                number of data point being queried per AL iter
        output_dir:
                where to save all the results
        save_models:
                whether we want to save the models' parameters (.txt)
        save_figs:
                whether we want to plot the models or not (evaluations will be plotted anyway)
        save_every_N_step: if save_figs is True or save_models is True,
                the models will be plotted/saved every specified number of AL iters
        display_figs:
                show the plot or not
        save_iv: for data exclusive sparse models,
                we can save the information of inducing points in all AL iters if needed
        template_model: whether we want pretrain main models, 
                if provided, this model will be trained before AL and all main models take the parameters from this one
        template_safety_model: whether we want pretrain safety models, 
                if provided, this model will be trained before AL and all safety models take the parameters from this one
        """
        self.seed = seed
        self.exp_idx = exp_idx
        self.data_manager = data_manager
        self.get_models(model_dictionary, safety_model_dictionary)
        
        self.num_init_data = np.sum(self.models[self.model_names[0]].history['data_history']==0)
        
        self.optimizer_args = optimizer_args
        self.fixed_initial_dataset = fixed_initial_dataset
        self.iter_num_AL = iter_num_AL
        self.query_policy = query_policy
        self.query_num = query_num
        self.output_dir = output_dir
        self.save_models = save_models
        self.save_figs = save_figs
        self.save_every_N_step = save_every_N_step
        self.display_figs = display_figs
        self.plot_kernels = plot_kernels
        self.save_iv = save_iv
        self.template_model = template_model
        self.template_safety_model = template_safety_model
        self.POO = partially_observed_output
        self.full_GP = isinstance(model_dictionary[self.model_names[0]], myMOGPR)
        self.BI = bayesian_inference
    
    def run_pipeline(self):
        np.random.seed(self.seed)
        if self.iter_num_AL > 0:
            self.exp_type = 'AL'
            self.run_AL()
        elif self.iter_num_AL == 0:
            self.exp_type = 'model_test'
            self.run_model_test()
    
    def run_AL(self):
        print("\n##############################################"+\
              "\n######                                  ######"+\
              "\n######       Experiment %7d         ######"%(self.exp_idx)+\
              "\n######                                  ######"+\
              "\n######     preparing the experiment     ######")
        # check if output_dir exists, create dirs if not
        self.check_path_exist()
        # prepare result_place_holders
        self.create_result_place_holder(record_pred_mean=True)
        # save experiment setup
        self.save_exp_setup()
        # pretrain if needed
        if not self.template_model is None and not self.BI:
            print("######       pretrain main models       ######")
            self.model_pretrain(self.template_model, **self.optimizer_args)
        if not self.template_safety_model is None and not self.BI:
            print("######       pretrain safety models     ######")
            self.safety_model_pretrain(self.template_safety_model, **self.optimizer_args)
        
        print("######                                  ######"+\
              "\n##############################################")
        
        # run our safe AL
        for i in range(self.iter_num_AL):
            print("\n##############################################"+\
                  "\n######                                  ######"+\
                  "\n######       Experiment %7d         ######"%(self.exp_idx)+\
                  "\n######                                  ######"+\
                  "\n######       iteration  %4d/%4d       ######"%(i+1, self.iter_num_AL)+\
                  "\n######                                  ######"+\
                  "\n######             Training             ######"+\
                  "\n######            main models           ######")
            times = self.model_train(**self.optimizer_args)
            self.print_times(times)
            
            print("######            safety models         ######")
            times = self.safety_model_train(**self.optimizer_args)
            self.print_times(times)
            
            print("######                                  ######"+\
                "\n######       evaluating the models      ######")
            times = self.model_evaluation(i)
            self.print_times(times)
            
            # save what we want
            if i % self.save_every_N_step == 0 or i == self.iter_num_AL-1:
                # plot models if needed
                if self.save_figs or self.display_figs:
                    print("######                                  ######"+\
                        "\n######         plotting models          ######")
                    times = self.model_plot(self.exp_idx, i, self.save_figs)
                    self.print_times(times)
                # plot kernel if needed
                if self.plot_kernels or self.display_figs:
                    print("######         plotting kernels         ######")
                    times = self.kernel_plot(self.exp_idx, i, self.save_figs)
                    self.print_times(times)
                    print("######         plotting kernel diffs    ######")
                    times = self.kernel_diff_plot(self.exp_idx, i, self.save_figs)
                    self.print_times(times)
                # save models parameters if needed
                if self.save_models:
                    print("######                                  ######"+\
                        "\n######         saving models            ######")
                    if self.BI:
                        self.save_all_models_to_histogram(self.exp_idx, i)
                    else:
                        self.save_all_models_to_txt(self.exp_idx, i)
                        self.save_kernel_pars(self.exp_idx, i)
            
            # save inducing points if needed
            if self.save_iv:
                for name in self.model_names:
                    self.save_iv_info(name, self.exp_idx, i)
            if i==self.iter_num_AL-1:
                print("######                                  ######"+\
                    "\n##############################################\n")
                break
            # query points
            print("######                                  ######"+\
                "\n######      determine safe points       ######")
            safety = self.determine_safe_points()
            print("######                                  ######"+\
                "\n######  query points / update datasets  ######")
            times = self.update_datasets(*safety)
            self.print_times(times)
            
            print("######                                  ######"+\
                "\n##############################################\n")
        # AL loop ends, save results
        print("\n##############################################"+\
              "\n######                                  ######"+\
              "\n######       Experiment %7d         ######"%(self.exp_idx)+\
              "\n######                                  ######"+\
              "\n######   save model evaluation result   ######")
        self.save_evaluations(self.exp_idx)
        print("######                                  ######"+\
            "\n######         experiment done!!!       ######"+\
            "\n######                                  ######"+\
            "\n##############################################\n")
    
    def run_model_test(self):
        print("\n##############################################"+\
              "\n######                                  ######"+\
              "\n######       Experiment %7d         ######"%(self.exp_idx)+\
              "\n######                                  ######"+\
              "\n######     preparing the experiment     ######")
        # check if output_dir exists, create dirs if not
        self.check_path_exist()
        # prepare result_place_holders
        self.create_result_place_holder(record_pred_mean=False)
        # save experiment setup
        self.save_exp_setup()
        # pretrain if needed
        if not self.template_model is None and not self.BI:
            all_names = self.model_names
            pretrained_name = []
            for name in all_names:
                pretrained_name.append(name + '_pretrained')
                self.models[name + '_pretrained'] = deepcopy(self.models[name])
                self.safety_models[name + '_pretrained'] = deepcopy(self.safety_models[name])
            all_names.extend(pretrained_name)
            
            print("######       pretrain main models       ######")
            self.model_names = pretrained_name # so the later method trained only pretrained models
            
            self.model_pretrain(self.template_model, **self.optimizer_args)
            if not self.template_safety_model is None:
                print("######       pretrain safety models     ######")
                self.safety_model_pretrain(self.template_safety_model, **self.optimizer_args)
            
            self.model_names = all_names

        
        """
        pseudo code
        
        1.train
        
        2.save model_objs (for pretrained models, save pre- & post-trained objs)
        3.save RMSE training, RMSE test, log dens training, log dens test
        4.plot if needed
        
        5.pretrain & train if needed (then do 2,3,4)
        """
        print("######                                  ######"+\
              "\n######  saving objectives  (pre-train)  ######")
        self.record_model_objs(add_name='_pre')
        print("######                                  ######"+\
              "\n######             Training             ######"+\
              "\n######            main models           ######")
        times = self.model_train(**self.optimizer_args)
        self.print_times(times)
        print("######            safety models         ######")
        times = self.safety_model_train(**self.optimizer_args)
        self.print_times(times)
        print("######                                  ######"+\
              "\n######  saving objectives (post-train)  ######")
        self.record_model_objs(add_name='_post')
        print("######                                  ######"+\
              "\n######       evaluating the models      ######")
        times = self.model_evaluation(-1) # in this case any input is fine
        self.print_times(times)
        
        # save what we want
        # plot models if needed
        if self.save_figs or self.display_figs:
            print("######                                  ######"+\
                  "\n######         plotting models          ######")
            times = self.model_plot(self.exp_idx, -1, self.save_figs) # replace -1 by any values is fine
            self.print_times(times)
        # plot kernel if needed
        if self.plot_kernels or self.display_figs:
            print("######         plotting kernels         ######")
            times = self.kernel_plot(self.exp_idx, -1, self.save_figs) # replace -1 by any values is fine
            self.print_times(times)
        
        # save models parameters if needed
        if self.save_models:
            print("######                                  ######"+\
                  "\n######         saving models            ######")
            if self.BI:
                self.save_all_models_to_histogram(self.exp_idx, -1) # replace -1 by any values is fine
            else:
                self.save_all_models_to_txt(self.exp_idx, -1)
                self.save_kernel_pars(self.exp_idx, -1) # replace -1 by any values is fine
        
        
        
        
        # AL loop ends, save results
        print("######                                  ######"+\
              "\n######   save model evaluation result   ######")
        self.save_evaluations(self.exp_idx)
        print("######                                  ######"+\
            "\n######         experiment done!!!       ######"+\
            "\n######                                  ######"+\
            "\n##############################################\n")
    
    def print_times(self, time_dict):
        for name, t in time_dict.items():
            print("###### %-20s: %7.3f(s) ######"%(name, t))
        
    def check_path_exist(self):
        if self.exp_type =='AL':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.output_dir, "pkl_files", "individual_trials")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.output_dir, "models")).mkdir(parents=True, exist_ok=True)
        elif self.exp_type == 'model_test':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.output_dir, "individual_trials")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.output_dir, "plots")).mkdir(parents=True, exist_ok=True)
    
    def get_models(self, model_dict, safety_model_dict):
        self.model_names = list(model_dict.keys())
        self.models = model_dict
        self.safety_models = safety_model_dict
        self.data_inclusive = hasattr(model_dict[self.model_names[0]], 'data')
    
    def create_result_place_holder(self, record_pred_mean=False):
        
        used_y_ind = self.data_manager.used_y_ind
        
        if record_pred_mean:
            N_test = self.data_manager.raw_data_test["Y"].shape[0]
            self.pred_mean = result_placeholder(self.model_names, shape=[1, N_test, len(used_y_ind)])
        
        if self.exp_type == 'AL':
            place_holder_shape = [self.iter_num_AL, len(used_y_ind)]
            self.safety_summary = {
                name: pd.DataFrame(
                    np.zeros([0,4], dtype=int),
                    columns=['num_all_safe_return', 'num_really_safe', 'query_z', 'query_safe_bool']
                ) for name in self.model_names
            }
            self.RMSE_mean_training_safe = result_placeholder(self.model_names, shape=place_holder_shape)
            self.RMSE_mean_safe = result_placeholder(self.model_names, shape=place_holder_shape)
        
        elif self.exp_type == 'model_test':
            place_holder_shape = [len(used_y_ind)]
            self.model_objs = {}
            self.safety_model_objs = {}
        
        self.RMSE_mean_training = result_placeholder(self.model_names, shape=place_holder_shape)
        self.RMSE_mean = result_placeholder(self.model_names, shape=place_holder_shape)
        self.training_log_density = result_placeholder(self.model_names, shape=place_holder_shape)
        self.test_log_density = result_placeholder(self.model_names, shape=place_holder_shape)
    
    
    def save_exp_setup(self):
        NX_mtx = self.data_manager.NX_matrix
        pt1_inputs = self.data_manager.raw_data_training['pt1_inputs']
        pt1_outputs = self.data_manager.raw_data_training['pt1_outputs']
        used_y_ind = self.data_manager.used_y_ind
        used_z_ind = self.data_manager.used_z_ind
        Z_threshold = self.safety_models[self.model_names[0]].Z_threshold
        prob_threshold = self.safety_models[self.model_names[0]].prob_threshold
        with open(os.path.join(self.output_dir, 'experiment_setup.txt'), 'w') as fp:
            time_now = datetime.datetime.now(tz = tz)
            print(time_now.strftime('%Z (UTC%z)\n%Y.%b.%d  %A  %H:%M:%S\n'), file = fp)
            print("seed                               : %d"%(np.random.get_state()[1][0]), file=fp)
            # notice that this has to be called before any random funcs
            print("experiment index                   : %d"%(self.exp_idx), file=fp)
            print("data, NX matrix                    : \n%s"%(NX_mtx), file=fp)
            print("data, pt1 (input channels)         : %s"%(pt1_inputs), file=fp)
            print("data, pt1 (output channels)        : %s"%(pt1_outputs), file=fp)
            print("data, output idx for y (0-7)       : %s"%(used_y_ind), file=fp)
            print("data, output idx for z (0-7)       : %s"%(used_z_ind), file=fp)
            
            print(f"partially observed Y               : {self.POO}", file=fp)
            print(f"model, optimizer                   : {self.optimizer_args}", file=fp)
            if self.BI:
                print(f"model, HMC settings                : {self.models[self.model_names[0]].return_config()}", file=fp)
            print(f"are they full GPs                  : {self.full_GP}", file=fp)
            print(f"fully bayesian inference (full GP) : {self.BI}", file=fp)
            print(f"pretrain model(s)                  : {not self.template_model is None and not self.BI}", file=fp)
            print(f"pretrain safety model(s)           : {not self.template_safety_model is None and not self.BI}", file=fp)
            print("AL, num of initial training points : %d"%(self.num_init_data), file = fp)
            #if self.POO:
            #    print("        (notice: partially observed outputs have num = {num of x} * {output dimension})", file = fp)
            print("AL, fixed initial training set     : %s"%(self.fixed_initial_dataset), file = fp)
            print("AL, num of iterations              : %d"%(self.iter_num_AL), file = fp)
            print("AL, num of quering points          : %d"%(self.query_num), file = fp)
            print(f"AL, query strategy                : {self.query_policy}", file = fp)
            print("safety threshold (probability)     : <%f (with p>=%.3f)"%(Z_threshold, prob_threshold), file = fp)
        
    def record_model_objs(self, add_name = ''):
        
        for name in self.model_names:
            if self.data_inclusive:
                data = self.models[name].data
                data_safety = self.safety_models[name].data
            else:
                key = name[:-11] if name.endswith('_pretrained') else name        
                _, U, Y, Z = self.data_manager.return_training_data_tuple(key)
                data = (U, Y)
                data_safety = (U, Z)
            
            loss = self.models[name].return_loss(data)
            self.model_objs[name + add_name] = loss().numpy()
            
            loss = self.safety_models[name].return_loss(data_safety)
            self.safety_model_objs[name + add_name] = loss().numpy()
        
    
    def model_pretrain(self, template_model, **optimizer_args):
        if self.data_inclusive:
            data = template_model.data
        else:
            name = self.model_names[0][:-11] if self.model_names[0].endswith('_pretrained') else self.model_names[0]
            _, U, Y, _ = self.data_manager.return_training_data_tuple(name)
            data = (U, Y)
        
        timer = template_model.training(data, **optimizer_args)
        template_model.history["training_time"].append(timer)
        
        for name in self.model_names:
            self.models[name].get_values_from_model(template_model)
        
    def safety_model_pretrain(self, template_model, **optimizer_args):
        if self.data_inclusive:
            data = template_model.data
        else:
            name = self.model_names[0][:-11] if self.model_names[0].endswith('_pretrained') else self.model_names[0]
            _, U, _, Z = self.data_manager.return_training_data_tuple(self.model_names[0])
            data = (U, Z)
        
        timer = template_model.training(data, **optimizer_args)
        template_model.history["training_time"].append(timer)
        
        for name in self.model_names:
            self.safety_models[name].get_values_from_model(template_model)
    
    def model_train(self, **optimizer_args):
        for name in self.model_names:
            if self.data_inclusive:
                data = self.models[name].data
            else:
                key = name[:-11] if name.endswith('_pretrained') else name
                _, U, Y, _ = self.data_manager.return_training_data_tuple(key)
                data = (U, Y)
            infer(self.models[name], BI=self.BI, best_training=False, data=data, **optimizer_args)

        times = {}
        for name in self.model_names:
            times[name] = self.models[name].history["training_time"][-1]
            
        return times
    
    def safety_model_train(self, **optimizer_args):
        for name in self.model_names:
            if self.data_inclusive:
                data = self.safety_models[name].data
            else:
                key = name[:-11] if name.endswith('_pretrained') else name
                _, U, _, Z = self.data_manager.return_training_data_tuple(key)
                data = (U, Z)

            infer(self.safety_models[name], BI = self.BI, best_training = False, data = data, ** optimizer_args)
        
        times = {}
        for name in self.model_names:
            times[name] = self.safety_models[name].history["training_time"][-1]
        
        return times

    def model_evaluation(self, iter_idx):
        
        times = {}
        _, U_training_pool, Y_training_pool, Z_training_pool = self.data_manager.training_pool
        _, U_test_pool, Y_test_pool, Z_test_pool = self.data_manager.test_pool
        
        if self.exp_type == 'AL':
            Z_threshold = self.safety_models[self.model_names[0]].Z_threshold
            mask_safe_training = (Z_training_pool <= Z_threshold).reshape(-1)
            # Z_training_pool is supposed to be of shape [N_tr, 1]
            mask_safe_test = (Z_test_pool <= Z_threshold).reshape(-1)
            # Z_test_pool is supposed to be of shape [N_te, 1]

        for name in self.model_names:
            timer = -time.perf_counter()
            
            mu, _ = self.models[name].predict_f(U_training_pool)
            rmse_training = np.sqrt(np.mean(
                        np.power(Y_training_pool - mu.numpy(), 2),
                        axis=-2
                        ))
            if self.exp_type == 'AL':
                rmse_training_safe = np.sqrt(np.mean(
                        np.power(Y_training_pool[mask_safe_training] - mu.numpy()[mask_safe_training], 2),
                        axis=-2
                        ))
            
            mu, _ = self.models[name].predict_f(U_test_pool)
            if hasattr(self, 'pred_mean'):
                self.pred_mean[name][..., :, :] = mu.numpy()
            rmse_test = np.sqrt(np.mean(
                        np.power(Y_test_pool - mu.numpy(), 2),
                        axis=-2
                        ))
            if self.exp_type == 'AL':
                rmse_test_safe = np.sqrt(np.mean(
                        np.power(Y_test_pool[mask_safe_test] - mu.numpy()[mask_safe_test], 2),
                        axis=-2
                        ))

            log_dens_training = self.models[name].predict_log_density_full_output((U_training_pool, Y_training_pool)).numpy().sum(axis=0)
            log_dens_test = self.models[name].predict_log_density_full_output((U_test_pool, Y_test_pool)).numpy().sum(axis=0)
            
            timer += time.perf_counter()
            
            times[name] = timer
        
            if self.exp_type == 'AL':
                
                self.RMSE_mean_training[name][iter_idx, :] = rmse_training
                self.RMSE_mean_training_safe[name][iter_idx, :] = rmse_training_safe
                self.RMSE_mean_safe[name][iter_idx, :] = rmse_test_safe
                self.RMSE_mean[name][iter_idx, :] = rmse_test
                self.training_log_density[name][iter_idx, :] = log_dens_training
                self.test_log_density[name][iter_idx, :] = log_dens_test
                
            elif self.exp_type == 'model_test':
                
                self.RMSE_mean_training[name] = rmse_training
                self.RMSE_mean[name] = rmse_test
                self.training_log_density[name] = log_dens_training
                self.test_log_density[name] = log_dens_test
            
        
        return times
    
    def model_plot(self, exp_idx, iter_idx, save_figs=False):
        times = {}
        pool_tr = self.data_manager.training_pool
        pool_te = self.data_manager.test_pool
        channel_names = self.data_manager.name_tuple
        NX_mtx = self.data_manager.NX_matrix
        
        for name in self.model_names:
            timer = -time.perf_counter()
            
            if self.exp_type == 'AL':
                fig_name = os.path.join(
                    self.output_dir,
                    "models",
                    f"exp_{exp_idx}_{name}_iter{iter_idx}.jpg"
                    )
            elif self.exp_type == 'model_test':
                fig_name = os.path.join(
                    self.output_dir,
                    "plots",
                    f"data_num_{self.num_init_data}_model_{name}_exp_{exp_idx}.jpg"
                    )
            
            _ = plot_OWEO(
                self.models[name], self.safety_models[name],
                pool_tr, pool_te, channel_names, NX_mtx,
                global_title = name,
                savefile=[save_figs, fig_name],
                display_fig=self.display_figs
                )
            
            timer += time.perf_counter()
            
            times[name] = timer
        
        return times
    
    def kernel_plot(self, exp_idx, iter_idx, save_figs=False):
        times = {}
        
        for name in self.model_names:
            timer = -time.perf_counter()
            
            if self.data_inclusive:
                U,_ = self.models[name].data
            else:
                key = name[:-11] if name.endswith('_pretrained') else name
                _, U, _, _ = self.data_manager.return_training_data_tuple(key)
            
            
            if self.exp_type == 'AL':
                fig_name = os.path.join(
                    self.output_dir,
                    "models",
                    f"exp_{exp_idx}_{name}_iter{iter_idx}_kernel.jpg"
                    )
            elif self.exp_type == 'model_test':
                fig_name = os.path.join(
                    self.output_dir,
                    "plots",
                    f"data_num_{self.num_init_data}_kernel_{name}_exp_{exp_idx}.jpg"
                    )
            
            
            _ = plot_kernel(
                self.models[name].kernel(U, U, full_cov=True).numpy(),
                title = name+f", obs. var: {self.models[name].likelihood.variance.numpy()}",
                savefile=[save_figs, fig_name],
                display_fig=self.display_figs)
            
            timer += time.perf_counter()
            
            times[name] = timer
        
        return times
    
    def save_kernel_pars(self, exp_idx, iter_idx):
        
        for name, model in self.models.items():
            if self.exp_type == 'AL':
                fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"kernel_{name}_exp{exp_idx}_iter{iter_idx}.pkl")
            elif self.exp_type == 'model_test':
                fullpath = os.path.join(self.output_dir, "individual_trials", f"data_num_{self.num_init_data}_kernel_{name}_exp{exp_idx}.pkl")
            
            savefile(fullpath, gpflow.utilities.read_values(model.kernel))
        
    
    def kernel_diff_plot(self, exp_idx, iter_idx, save_figs=False):
        times = {}
        
        for j, name in enumerate(self.model_names):
            
            if self.data_inclusive:
                U,_ = self.models[name].data
            else:
                key = name[:-11] if name.endswith('_pretrained') else name
                _, U, _, _ = self.data_manager.return_training_data_tuple(key)
            
            if j == 0:
                kernel_base = self.models[name].kernel(U, U, full_cov=True).numpy()
                name_base = name
                continue
            else:
                kernel_matx = self.models[name].kernel(U, U, full_cov=True).numpy()
            timer = -time.perf_counter()
            
            _ = plot_kernel(
                np.abs(kernel_matx - kernel_base),
                title = f'absolute difference of {name} & {name_base}',
                savefile=[save_figs, os.path.join(self.output_dir, "models", f"exp_{exp_idx}_{name}_iter{iter_idx}_kernel_diff.jpg")],
                display_fig=self.display_figs)
            
            timer += time.perf_counter()
            
            times[name] = timer
        
        return times
    
    def determine_safe_points(self):
        safe_pools = {}
        safe_args = {}
        _, U_training_pool, _, _ = self.data_manager.training_pool
        y_dim = self.data_manager.return_dimensions('Y')
        
        for name in self.model_names:
            # if the outputs are partially observed, shape of
            # self.models[name].history['data_args']
            # is [2, N], where the 2nd row is indices of tasks
            used_args = self.models[name].history['data_args']
            if self.POO: # partially observed output
                # in this case np.shape(used_args) == [2, N]
                used_args_base, count = np.unique(used_args[0,:], return_counts=True)
                partially_queried_args = used_args_base[count < y_dim]
                used_args = used_args_base[count==y_dim]
            
            # exclude used data
            excluded_args = np.hstack((
                self.data_manager.excluded_args,
                used_args
                ))
            # find safe points
            _, arg_safe = self.safety_models[name].return_safe_points(U_training_pool, excluded_args, return_index=True, safe_above_threshold=False)
            safe_args[name] = arg_safe
            safe_pools[name] = self.data_manager.training_data_selection(safe_args[name])
            # if the outputs are partially observed, some outputs may be partially queried
            # so be careful about the remaining pool
            if self.POO:
                X, U, Y, Z = safe_pools[name]
                used_args = self.models[name].history['data_args']
                partially_queried_args = partially_queried_args[np.in1d(partially_queried_args, arg_safe)]
                non_queried_args = arg_safe[~np.in1d(arg_safe, partially_queried_args)]
                
                part_args_used_task = args_partially_queried_matrix(
                    used_args[:, np.in1d(used_args[0,:], partially_queried_args)].reshape([2,-1]),
                    y_dim
                    )
                
                queried_mtx = part_args_used_task.append(
                    pd.DataFrame(np.zeros([len(non_queried_args), y_dim], dtype=bool), index=non_queried_args)
                    ).loc[arg_safe, :].values
                
                Y[queried_mtx] = np.nan
                safe_pools[name] = (X, U, Y, Z)
                
        return safe_pools, safe_args
    
    def update_datasets(self, safe_pools, safe_args):
        times = {}
        for ind, name in enumerate(self.model_names):
            
            X_safe, U_safe, Y_safe, Z_safe = safe_pools[name]
            Dpool = (U_safe, Y_safe)

            # query point(s)
            Dnew, arg_new = self.models[name].query_points(
                Dpool, num_return = self.query_num,
                return_index=True, original_args=safe_args[name],
                full_task_query = not self.POO,
                **self.query_policy[name]
                # when global_entropy_first==False, full_output_cov is a useless parameter
                )

            _, _, _, Z_new = \
                self.data_manager.training_data_selection(arg_new)

            self.safety_summary[name] = self.safety_summary[name].append(
                pd.DataFrame([[
                    Z_safe.shape[0],  # the total points returned by model as safe
                    np.sum(Z_safe <= self.safety_models[name].Z_threshold),  # how many are actually safe
                    Z_new,
                    Z_new <= self.safety_models[name].Z_threshold
                ]], columns=self.safety_summary[name].columns.values
                )
            )

            if self.data_inclusive:
                self.models[name].update_dataset(Dnew, data_args=arg_new)
                self.safety_models[name].update_dataset((Dnew[0], Z_new), data_args=arg_new)
            else:
                self.data_manager.training_data_update(arg_new, name)
                self.safety_models[name].history_update(arg_new)
            
            times[name] = self.models[name].history["point_selection_time"][-1]
            
        return times
            
    def save_iv_info(self, model_name, exp_idx, iter_idx):
        r"""
        only use this functions for data exclusive models
        """
        if self.data_inclusive:
            raise ValueError("currently not supporting data inclusive models")
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"ivs_{model_name}_exp{exp_idx}_iter{iter_idx}.pkl")
        savefile(fullpath, self.models[model_name].inducing_variable.inducing_variable.Z.numpy(), mode='wb')
        
        _, U_training_pool, _, _ = self.data_manager.training_pool
        training_pool_mean_dis = self.models[model_name].distance_to_iv(U_training_pool).numpy().mean(axis=0)
        used_training_dis = self.models[model_name].distance_to_iv(self.data_manager.U[model_name]).numpy()
        # notice that self.data_manager.U only exists in data exclusive experiments
        
        used_training_dis[:-self.query_num,...]
        
        open_mode = 'w' if iter_idx==0 else 'a'
        with open(os.path.join(self.output_dir, f'iv_info_{model_name}.txt'), open_mode) as fp:
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            time_now = datetime.datetime.now(tz = tz)
            print(time_now.strftime('%Z (UTC%z)\n%Y.%b.%d  %A  %H:%M:%S\n'), file = fp)
            print(f"AL iter: {iter_idx}\n mean distance of full training pool to ivs: {training_pool_mean_dis}", file=fp)
            print(f'inducing points: \n{self.models[model_name].inducing_variable.inducing_variable.Z.numpy()}', file=fp)
            print(f'distance of used training data to ivs:\n{used_training_dis[:-self.query_num,...].T}', file=fp)
            print(f'distance of queried data to ivs:\n{used_training_dis[-self.query_num:,...].T}', file=fp)
            print('\n\n\n', file=fp)

    def save_all_models_to_txt(self, exp_idx, iter_idx, save_ivs=False, save_variational_pars=False):
        # main models
        if self.exp_type == 'AL':
            path = os.path.join(
                self.output_dir, 'models',
                f'model_exp_{exp_idx}_iter_{iter_idx}_'
                )
        elif self.exp_type == 'model_test':
            path = os.path.join(
                self.output_dir, 'individual_trials',
                f'model_data_num_{self.num_init_data}_exp_{exp_idx}_'
                )    
        for name, model in self.models.items():
            save_model(path+str(name)+'.txt', model, 'w', save_ivs, save_variational_pars)
        
        # safety_models
        if self.exp_type == 'AL':
            path = os.path.join(
                self.output_dir, 'models',
                f'model_safety_exp_{exp_idx}_iter_{iter_idx}_'
                )
        elif self.exp_type == 'model_test':
            path = os.path.join(
                self.output_dir, 'individual_trials',
                f'model_safety_data_num_{self.num_init_data}_exp_{exp_idx}_'
                )
        for name, model in self.safety_models.items():
            save_model(path+str(name)+'.txt', model, 'w', save_ivs, save_variational_pars)
        
    def save_all_models_to_histogram(self, exp_idx, iter_idx):
        # main models
        if self.exp_type == 'AL':
            path = os.path.join(
                self.output_dir, 'models',
                f'model_exp_{exp_idx}_iter_{iter_idx}_'
                )
        elif self.exp_type == 'model_test':
            path = os.path.join(
                self.output_dir, 'plots',
                f'model_data_num_{self.num_init_data}_exp_{exp_idx}_'
                )
        for name, model in self.models.items():
            uncon_d, con_d = model.return_all_model_samples()
            """
            plot_parameter_hist(
                uncon_d, title='unconstrained_parameters',
                savefile=[True, path + str(name) + "_hist_uncons.jpg"],
                display_fig=self.display_figs
                )
            """
            plot_parameter_hist(
                con_d, title='constrained_parameters',
                savefile=[True, path + str(name) + "_hist_cons.jpg"],
                display_fig=self.display_figs
                )
            with open(path + str(name) + '_history.txt', mode='w') as fp:
                for key, values in model.history.items():
                    if len(np.shape(values)) <= 1:
                        print(f'{key}: {values}', file=fp)
                    else:
                        print(f'{key}:\n{values}', file=fp)
        
        # safety models
        if self.exp_type == 'AL':
            path = os.path.join(
                self.output_dir, 'models',
                f'model_safety_exp_{exp_idx}_iter_{iter_idx}_'
                )
        elif self.exp_type == 'model_test':
            path = os.path.join(
                self.output_dir, 'plots',
                f'model_safety_data_num_{self.num_init_data}_exp_{exp_idx}_'
                )
        for name, model in self.safety_models.items():
            uncon_d, con_d = model.return_all_model_samples()
            """
            plot_parameter_hist(
                uncon_d, title='unconstrained_parameters',
                savefile=[True, path + str(name) + "_hist_uncons.jpg"],
                display_fig=self.display_figs
                )
            """
            plot_parameter_hist(
                con_d, title='constrained_parameters',
                savefile=[True, path + str(name) + "_hist_cons.jpg"],
                display_fig=self.display_figs
                )
            with open(path + str(name) + '_history.txt', mode='w') as fp:
                for key, values in model.history.items():
                    if len(np.shape(values)) <= 1:
                        print(f'{key}: {values}', file=fp)
                    else:
                        print(f'{key}:\n{values}', file=fp)
        
    def save_evaluations(self, exp_idx):
        
        if self.exp_type == 'AL':
            self.save_evaluations_AL(exp_idx=exp_idx)
        elif self.exp_type == 'model_test':
            self.save_evaluations_MT(exp_idx=exp_idx)
        
        
    def save_evaluations_AL(self, exp_idx):
        Training_time = {
                name: self.models[name].history["training_time"] for name in self.model_names
                }
        data_selection_time = {
                name: self.models[name].history["point_selection_time"] for name in self.model_names
                }
        safety_data = {
            name: self.safety_models[name].history["data_args"] for name in self.model_names
        }

        fullpath = os.path.join(self.output_dir, "pkl_files", "pred_ref.pkl")
        data_test = self.data_manager.raw_data_test
        used_y_ind = self.data_manager.used_y_ind
        savefile(fullpath, {'X': data_test["X"], 'True_Y': data_test["Y"][:, used_y_ind]}, mode='wb')
        
        if hasattr(self, 'pred_mean'):
            fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"pred_mean_exp{exp_idx}.pkl")
            savefile(fullpath, self.pred_mean, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"RMSE_mean_training_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean_training, mode='wb')
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"RMSE_mean_training_safe_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean_training_safe, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"RMSE_mean_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean, mode='wb')
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"RMSE_mean_safe_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean_safe, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"training_log_density_exp{exp_idx}.pkl")
        savefile(fullpath, self.training_log_density, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"test_log_density_exp{exp_idx}.pkl")
        savefile(fullpath, self.test_log_density, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"Training_time_exp{exp_idx}.pkl")
        savefile(fullpath, Training_time, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"data_selection_time_exp{exp_idx}.pkl")
        savefile(fullpath, data_selection_time, mode='wb')

        fullpath_noextension = os.path.join(self.output_dir, "pkl_files", "individual_trials", f"safety_summary_exp{exp_idx}")
        savefile(fullpath_noextension+'.pkl', self.safety_summary, mode='wb')
        with open(fullpath_noextension+'.txt', mode='w') as fp:
            self.print_safety_summary(fp)

        
    
    def save_evaluations_MT(self, exp_idx):
        Training_time = {
                name: self.models[name].history["training_time"] for name in self.model_names
                }
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"model_objs_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, self.model_objs, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"RMSE_mean_training_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean_training, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"RMSE_mean_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, self.RMSE_mean, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"training_log_density_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, self.training_log_density, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"test_log_density_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, self.test_log_density, mode='wb')
        
        fullpath = os.path.join(self.output_dir, "individual_trials", f"Training_time_data_num{self.num_init_data}_exp{exp_idx}.pkl")
        savefile(fullpath, Training_time, mode='wb')

    def print_safety_summary(self, file=sys.stdout):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 2000)

        for name in self.model_names:
            print(f'model: {name}', file=file)
            #print(f'      data_history   : {self.safety_models[name].history["data_history"]}', file=file)
            #print(f'      data_args      : {self.safety_models[name].history["data_args"]}', file=file)
            #print(f'   observation safe  : {None}', file=file)
            #print(f'   safe probability  : {None}\n', file=file)
            print(f'{self.safety_summary[name]}\n\n', file=file)


        
























