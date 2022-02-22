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
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from my_plot import *
from my_toolkit import str2bool, loadfile, savefile, collect_results, collection_filename, collect_all
from default_parameters import default_parameters_GPsamples

plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=16) 
plt.rc('legend',fontsize=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect exp results and plot.')
    
    pars = default_parameters_GPsamples()
    
    parser.add_argument('--dir', default= pars.output_dir, type=str, help="where are the results needed to plot")
    parser.add_argument('--display_figs', default= False, type=str2bool, nargs='?', const=True, help="whether the figures are shown")
    parser.add_argument('--save_figs', default= True, type=str2bool, nargs='?', const=True, help="whether the figures are saved")
    parser.add_argument('--plot_boxes', default= True, type=str2bool, nargs='?', const=True, help="whether the boxes (distribution) are shown")
    parser.add_argument('--plot_std_error', default= True, type=str2bool, nargs='?', const=True, help="whether the std (shadow) are shown")
    parser.add_argument('--significance_test', default= False, type=str2bool, nargs='?', const=True, help="whether the Wilcoxon signed-rank test is conducted")
    parser.add_argument('--plot_prediction', default=False, type=str2bool, nargs='?', const=True, help="")
    parser.add_argument('--plot_few_models', default=100, type=int, help="plot only the first specified number of models")
    
    args = parser.parse_args()
    
    display_figs = args.display_figs
    save_figs = args.save_figs
    plot_boxes = args.plot_boxes and not args.plot_std_error
    plot_std_error = args.plot_std_error
    st = args.significance_test
    
    path_list = glob.glob(os.path.join(args.dir + "*", "*"))
    
    for file_dir in path_list:
        
        print("##############################################" +\
              "\n### processing data at '%s'"%(file_dir))
        if not os.path.isdir(file_dir):
            print("### no result found"+\
                  "\n##############################################")
            continue
        
        
        # first obtain exp_idx, just to make sure the files has idx in order
        # this is optional, can also put exp_idx=None
        exp_idx = []
        for name in glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", "*Training_time*.pkl")):
            _, name = name.split('time_exp')
            idx, _ = name.split('.pkl')
            exp_idx.append(int(idx))
        exp_idx = np.sort(exp_idx)
        
        # Training_time
        print("### training time")
        Training_time = collect_all(os.path.join(file_dir, 'pkl_files'), 'Training_time', exp_idx=exp_idx)
        # data_selection_time
        print("### querying time")
        data_selection_time = collect_all(os.path.join(file_dir, 'pkl_files'), 'data_selection_time', exp_idx=exp_idx)
        # now we can plot times
        if True:
            _ = plot_time(Training_time, data_selection_time,
                          savefile=[save_figs, os.path.join(file_dir, 'times.jpg')],
                          display_fig=display_figs)
        
        if args.plot_prediction:
            # load pred_ref, this is the ground truth (in toy dataset) & test_data (OWEO dataset)
            print("### test observations")
            pred_ref = loadfile(os.path.join(file_dir, "pkl_files", "pred_ref.pkl"), mode='rb')
            # pred_mean
            print("### function pridictions (mean)")
            pred_mean = collect_all(os.path.join(file_dir, 'pkl_files'), 'pred_mean', exp_idx=exp_idx)
        else:
            pred_ref = pred_mean = None
        # test_data, log_density
        print("### log densities (of test data)")
        test_log_density = collect_all(os.path.join(file_dir, 'pkl_files'), 'test_log_density', exp_idx=exp_idx)
        # test_data, RMSE_mean
        print("### RMSE_test")
        RMSE_mean = collect_all(os.path.join(file_dir, 'pkl_files'), 'RMSE_mean', exp_idx=exp_idx)
        # now plot results
        if True:
            _ = plot_RMSE_log_density(RMSE_mean, test_log_density, pred_mean, pred_ref,
                                      ylim = np.array([[[0.0, 1.3],
                                                        [0.0, 1.3],
                                                        [0.0, 1.3],
                                                        [0.0, 1.3],
                                                        [0.0, 1.3]]]),
                                      plot_boxes = plot_boxes, plot_std_error=plot_std_error,
                                      significance_test = st,
                                      saturation_values = np.array([[0.4, 0.4, 0.4, 0.4, 0.4]]),
                                      plot_model = args.plot_few_models,
                                      savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean.jpg')],
                                      display_fig=display_figs)
            
            _ = plot_RMSE_log_density_global(
                RMSE_mean, test_log_density, ylim = np.array([[[0.0, 1.3]]]),
                plot_boxes = plot_boxes, plot_std_error=plot_std_error,
                significance_test = st,
                saturation_values = np.array([[0.4]]),
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_global.jpg')],
                display_fig=display_figs
                )
            
            _ = plot_RMSE_log_density_diff(RMSE_mean, test_log_density,
                                           plot_boxes = plot_boxes,
                                           savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_diff.jpg')],
                                           display_fig=display_figs)
            
            _ = plot_RMSE_log_density_global_diff(
                RMSE_mean, test_log_density,
                plot_boxes = plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_global_diff.jpg')],
                display_fig=display_figs
                )
        
        # training_data, log_density
        print("### log densities (of training data)")
        training_log_density = collect_all(os.path.join(file_dir, 'pkl_files'), 'training_log_density', exp_idx=exp_idx)
        # training_data, RMSE_mean
        print("### RMSE_training")
        RMSE_mean_training = collect_all(os.path.join(file_dir, 'pkl_files'), 'RMSE_mean_training', exp_idx=exp_idx)
        # now plot mean_based results
        _ = plot_RMSE_log_density(RMSE_mean_training, training_log_density,
                                  ylim = np.array([[[0.0, 1.3],
                                                    [0.0, 1.3],
                                                    [0.0, 1.3],
                                                    [0.0, 1.3],
                                                    [0.0, 1.3]]]),
                                  plot_boxes = plot_boxes, plot_std_error=plot_std_error,
                                  significance_test = st,
                                  saturation_values = np.array([[0.4, 0.4, 0.4, 0.4, 0.4]]),
                                  plot_model = args.plot_few_models,
                                  savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_training.jpg')],
                                  display_fig=display_figs)
        
        _ = plot_RMSE_log_density_global(
            RMSE_mean_training, training_log_density,
            ylim = np.array([[[0.0, 1.3]]]),
            plot_boxes = plot_boxes, plot_std_error=plot_std_error,
            significance_test = st,
            saturation_values = np.array([[0.4]]),
            plot_model = args.plot_few_models,
            savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_training_global.jpg')],
            display_fig=display_figs
            )
        
        _ = plot_RMSE_log_density_diff(RMSE_mean_training, training_log_density,
                                       plot_boxes = plot_boxes,
                                       savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_training_diff.jpg')],
                                       display_fig=display_figs)
        
        _ = plot_RMSE_log_density_global_diff(
            RMSE_mean_training, training_log_density,
            plot_boxes = plot_boxes,
            savefile=[save_figs, os.path.join(file_dir, 'RMSE_mean_training_global_diff.jpg')],
            display_fig=display_figs)
        
        
        print("##############################################")
    
    print("##############"+\
          "\n### Done! ###"+\
          "\n### Done! ###"+\
          "\n### Done! ###"+\
          "\n#############")
    
    
    
    













