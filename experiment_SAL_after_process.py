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
from default_parameters import default_parameters_OWEO

plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=16) 
plt.rc('legend',fontsize=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect exp results and plot.')
    
    pars = default_parameters_OWEO()
    
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
    
    path_list = glob.glob(os.path.join(args.dir, "*", "AL*"))
    path_list.extend( glob.glob(os.path.join(args.dir + '_POO', "*", "AL*")) )
    
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
            pred_test_ref = loadfile(os.path.join(file_dir, "pkl_files", "pred_ref.pkl"), mode='rb')
            # pred_mean
            print("### function pridictions (mean)")
            pred_test = collect_all(os.path.join(file_dir, 'pkl_files'), 'pred_mean', exp_idx=exp_idx)
        else:
            pred_test_ref = pred_test = None
        
        for j in range(4):
            if j <= 1:
                # test_data
                if j == 0:
                    print("### log densities (of test data)")
                    log_density = collect_all(os.path.join(file_dir, 'pkl_files'), 'test_log_density', exp_idx=exp_idx)
                    print("### RMSE_test")
                    key = 'RMSE_mean'
                    pred = pred_test
                    pred_ref = pred_test_ref
                else:
                    log_density = None
                    print("### RMSE_test_safe")
                    key = 'RMSE_mean_safe'
                    pred = None
                    pred_ref = None
                RMSE = collect_all(os.path.join(file_dir, 'pkl_files'), key, exp_idx=exp_idx)

                ylim_ind = np.array([[[0.8, 1.5], [0.6, 1.2]]])# for HC & O2
                ylim_glob = np.array([[[0.7, 1.3]]])
                #ylim_ind = np.array([[[0.2, 1.2], [0.8, 1.5]]])# for CO2 & HC
                #ylim_glob = np.array([[[0.6, 1.3]]])
                satur_values_ind = np.array([[0.75, 0.45]])
                satur_values_glob = np.array([[1.0]])
            elif j <= 3:
                if j == 2:
                    print("### log densities (of training data)")
                    log_density = collect_all(os.path.join(file_dir, 'pkl_files'), 'training_log_density', exp_idx=exp_idx)
                    # training_data, RMSE_mean
                    print("### RMSE_training")
                    key = 'RMSE_mean_training'
                else:
                    log_density = None
                    print("### RMSE_training_safe")
                    key = 'RMSE_mean_training_safe'
                RMSE = collect_all(os.path.join(file_dir, 'pkl_files'), key, exp_idx=exp_idx)
                pred = None
                pred_ref = None
                ylim_ind = np.array([[[0.4, 1.0], [0.2, 1.0]]])# for HC & O2
                ylim_glob = np.array([[[0.3, 1.0]]])
                #ylim_ind = np.array([[[0.2, 1.0], [0.5, 1.0]]])# for CO2 & HC
                #ylim_glob = np.array([[[0.2, 1.0]]])
                satur_values_ind = np.array([[0.4, 0.3]])
                satur_values_glob = np.array([[0.3]])

            # now plot results
            _ = plot_RMSE_log_density(RMSE, log_density, pred, pred_ref,
                                      ylim = ylim_ind,
                                      plot_boxes = plot_boxes, plot_std_error=plot_std_error,
                                      significance_test = st,
                                      saturation_values = satur_values_ind,
                                      plot_model = args.plot_few_models,
                                      savefile=[save_figs, os.path.join(file_dir, key+'.jpg')],
                                      display_fig=display_figs)
            
            _ = plot_RMSE_log_density_global(
                RMSE, log_density, ylim = ylim_glob,
                plot_boxes = plot_boxes, plot_std_error=plot_std_error,
                significance_test = st,
                saturation_values = satur_values_glob,
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, key + '_global.jpg')],
                display_fig=display_figs
                )
            
            _ = plot_RMSE_log_density_diff(RMSE, log_density,
                                           plot_boxes = plot_boxes,
                                           savefile=[save_figs, os.path.join(file_dir, key+'_diff.jpg')],
                                           display_fig=display_figs)
            
            _ = plot_RMSE_log_density_global_diff(
                RMSE, log_density,
                plot_boxes = plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, key+'_global_diff.jpg')],
                display_fig=display_figs
                )

        
        print("##############################################")
    
    print("##############"+\
          "\n### Done! ###"+\
          "\n### Done! ###"+\
          "\n### Done! ###"+\
          "\n#############")
    
    
    
    













