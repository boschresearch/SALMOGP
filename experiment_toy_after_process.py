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
import pickle
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from my_plot import *
from my_toolkit import str2bool, collect_results, loadfile, savefile
from default_parameters import default_parameters_toy

import platform
if platform.system().lower() == 'linux':
    from matplotlib import use
    use('Agg')

plt.rcParams.update({'font.size': 22})
plt.rc('xtick', labelsize=16) 
plt.rc('legend',fontsize=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect exp results and plot.')
    
    pars = default_parameters_toy()
    
    parser.add_argument('--dir', default= pars.output_dir, type=str, help="where are the results needed to plot")
    parser.add_argument('--data_noise_std', default= pars.data_noise_std, nargs=2, type=float, help=f"default={pars.data_noise_std}, standard deviation of observations, 2 output dims")
    parser.add_argument('--display_figs', default= False, type=str2bool, nargs='?', const=True, help="whether the figures are shown")
    parser.add_argument('--save_figs', default= True, type=str2bool, nargs='?', const=True, help="whether the figures are saved")
    parser.add_argument('--plot_boxes', default=True, type=str2bool, nargs='?', const=True, help="")
    parser.add_argument('--plot_std_error', default= True, type=str2bool, nargs='?', const=True, help="whether the std (shadow) are shown")
    parser.add_argument('--significance_test', default=False, type=str2bool, nargs='?', const=True, help="")
    parser.add_argument('--plot_prediction', default=False, type=str2bool, nargs='?', const=True, help="")
    parser.add_argument('--plot_few_models', default=100, type=int, help="plot only the first specified number of models")
    args = parser.parse_args()
    
    data_noise_std = np.array(args.data_noise_std)
    plot_boxes = args.plot_boxes and not args.plot_std_error
    plot_std_error = args.plot_std_error
    display_figs = args.display_figs
    save_figs = args.save_figs
    
    path_list = glob.glob(os.path.join(args.dir + '*', '*'))
    
    for file_dir in path_list:
        print("### processing data at '%s'"%(file_dir))
        if args.plot_prediction:
            print("### predictions")
            # load pred_ref, this is the ground truth in toy dataset
            files = glob.glob(os.path.join(file_dir, "pkl_files", "*pred_ref*.pkl"))
            pred_ref = loadfile(files[0], mode = "rb")
        
            # pred_mean
            pred_mean = collect_results(glob.glob(
                os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_mean*.pkl")
                ))
            savefile(
                os.path.join(
                    file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_mean.pkl"
                    ), 
                pred_mean, mode = "wb"
                )
            # pred_draw & RMSE_draw
            pred_draw = collect_results(glob.glob(
                os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_draw*.pkl")
                ))
            savefile(
                os.path.join(
                    file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_pred_draw.pkl"
                    ),
                pred_draw, mode = "wb"
                )
        else:
            pred_ref = pred_mean = pred_draw = None
        # Training_time
        print("### training time")
        files = glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_Training_time*.pkl"))
        Training_time = collect_results(files)
        savefile(
            os.path.join(file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_Training_time.pkl"),
            Training_time, mode='wb'
            )
        # data_selection_time
        print("### querying time")
        files = glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_data_selection_time*.pkl"))
        data_selection_time = collect_results(files)
        savefile(
            os.path.join(file_dir, 'pkl_files', f'data_std{data_noise_std[0]}&{data_noise_std[1]}_data_selection_time.pkl'),
            data_selection_time, mode = "wb"
            )
        # now we can plot times
        try:
            _ = plot_time(Training_time, data_selection_time,
                      savefile=[save_figs, os.path.join(file_dir, f'data_std{data_noise_std[0]}&{data_noise_std[1]}_times.jpg')], display_fig=display_figs)
        except np.VisibleDeprecationWarning:
            pass
        
        # log_density
        print("### log densities")
        files = glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_test_log_density*.pkl"))
        test_log_density = collect_results(files)
        savefile(
            os.path.join(file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_test_log_density.pkl"),
            test_log_density, mode = "wb"
            )
        # RMSE_mean
        print("### RMSE (mean based)")
        files = glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean_exp*.pkl"))
        RMSE_mean = collect_results(files)
        savefile(
            os.path.join(file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean.pkl"),
            RMSE_mean, mode = "wb")
        # now plot mean_based results
        try:
            _ = plot_RMSE_log_density(
                RMSE_mean, test_log_density, pred_mean, pred_ref,
                ylim = np.tile([[[0.0, 1.0]]], [1, np.shape(list(RMSE_mean.items())[0][1])[2], 1]),
                plot_boxes=plot_boxes, plot_std_error=plot_std_error,
                significance_test=args.significance_test,
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean.jpg")],
                display_fig=display_figs
                )
            
            _ = plot_RMSE_log_density_global(
                RMSE_mean, test_log_density,
                ylim = np.array([[[0.0, 1.0]]]),
                plot_boxes=plot_boxes, plot_std_error=plot_std_error,
                significance_test=args.significance_test,
                saturation_values=np.array([[0.3]]),
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean_global.jpg")],
                display_fig=display_figs
                )
        except np.VisibleDeprecationWarning:
            pass
        
        try:
            _ = plot_RMSE_log_density_diff(
                RMSE_mean, test_log_density, plot_boxes=plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean_diff.jpg")],
                display_fig=display_figs)
            
            _ = plot_RMSE_log_density_global_diff(
                RMSE_mean, test_log_density, plot_boxes=plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_mean_global_diff.jpg")],
                display_fig=display_figs)
        except np.VisibleDeprecationWarning:
            pass
        
        print("### RMSE (draw based)")
        files = glob.glob(os.path.join(file_dir, "pkl_files", "individual_trials", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw*.pkl"))
        RMSE_draw = collect_results(files)
        savefile(
            os.path.join(file_dir, "pkl_files", f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw.pkl"),
            RMSE_draw, mode = "wb"
            )
        
        # now plot draw_based results
        try:
            _ = plot_RMSE_log_density(
                RMSE_draw, test_log_density, pred_draw, pred_ref,
                ylim = np.tile([[[0.0, 1.0]]], [1, np.shape(list(RMSE_draw.items())[0][1])[2], 1]),
                plot_boxes=plot_boxes, plot_std_error=plot_std_error,
                significance_test=args.significance_test,
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw.jpg")],
                display_fig=display_figs
                )
            
            _ = plot_RMSE_log_density_global(
                RMSE_mean, test_log_density,
                ylim = np.array([[[0.0, 1.0]]]),
                plot_boxes=plot_boxes, plot_std_error=plot_std_error,
                significance_test=args.significance_test,
                plot_model = args.plot_few_models,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw_global.jpg")],
                display_fig=display_figs
                )
        except np.VisibleDeprecationWarning:
            pass
        
        try:
            _ = plot_RMSE_log_density_diff(
                RMSE_draw, test_log_density, plot_boxes=plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw_diff.jpg")],
                display_fig=display_figs
                )
            
            _ = plot_RMSE_log_density_global_diff(
                RMSE_mean, test_log_density, plot_boxes=plot_boxes,
                savefile=[save_figs, os.path.join(file_dir, f"data_std{data_noise_std[0]}&{data_noise_std[1]}_RMSE_draw_global_diff.jpg")],
                display_fig=display_figs)
        except np.VisibleDeprecationWarning:
            pass
        
        print("### Done!")
    
    
    
    













