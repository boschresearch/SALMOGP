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
import pandas as pd
import os
import sys
import glob
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy import stats
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from my_toolkit import str2bool, loadfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect exp results and plot.')
    
    parser.add_argument('--dir', default=os.path.join('experimental_result', 'toy*', 'BGP'), type=str, help="default='experimental_result/toy*/BGP', where are the results needed to plot, * provides flexible search")
    parser.add_argument('--Z_threshold', default=1.0, type=float, help="")
    parser.add_argument('--safe_above_threshold', default=False, type=str2bool, nargs='?', const=True, help="default=False, whether the safety threshold is an upper bound (False) or lower bound (True)")

    args = parser.parse_args()

    path_list = glob.glob(args.dir)
    
    for file_path in path_list:
        print("##############################################" +\
              "\n### processing data at '%s'"%(file_path))
        
        summary_files = glob.glob(
            os.path.join(file_path, "pkl_files", "individual_trials", '*safety_summary_exp*.pkl')
        )
        num_exp = len(summary_files)
        
        if num_exp == 0:
            print("### no result found"+\
                  "\n##############################################")
            continue

        safety_summary = [loadfile(fdir, mode='rb') for fdir in summary_files]
        model_names = safety_summary[0].keys()

        columns = [model_names, ['mean', 'std.error', 'safe_query_ratio']]
        columns_names = pd.MultiIndex.from_product(columns, names=['model', 'quantity'])
        array = []
        safe_query = {}

        for name in model_names:
            pddfs = [safety_summary[i][name] for i in range(num_exp)] # this is a list of pd.DataFrame
            safe_prob = [
                (pddfs[i]['num_really_safe'].values / pddfs[i]['num_all_safe_return'].values).reshape([-1,1]) \
                for i in range(num_exp)
                ]
            query_z = [safety_summary[i][name]['query_z'].values.reshape([-1,1]) for i in range(num_exp)]

            safe_prob_array = np.concatenate(safe_prob, axis=-1) # [num_AL_iter, num_exp]
            query_z = np.concatenate(query_z, axis=-1) # [num_AL_iter, num_exp]

            sp_mean = safe_prob_array.mean(axis=-1).reshape([-1,1])
            sp_std_error = stats.sem(safe_prob_array, axis=-1).reshape([-1,1])
            really_safe = (query_z > args.Z_threshold) if args.safe_above_threshold else (query_z <= args.Z_threshold)
            safe_ratio = really_safe.sum(axis=-1).reshape([-1,1]) / num_exp

            array.append(sp_mean)
            array.append(sp_std_error)
            array.append(safe_ratio)

            safe_query[name] = really_safe.sum(axis=0) / query_z.shape[0]
        
        table = pd.DataFrame(np.concatenate(array, axis=-1), columns=columns_names)

        table.to_csv(
            os.path.join(file_path, 'safety_summary.csv'),
            float_format='%.5f',
            index=True
        )
        
        with open(os.path.join(file_path, 'safe_query.txt'), mode='w') as fp:
            print("Ratio of queries that are actually safe in each experiment", file=fp)
            for name in model_names:
                print(f"{name}: {safe_query[name]}\n   mean: {safe_query[name].mean()},  std.error: {stats.sem(safe_query[name])}\n", file=fp)
        
        