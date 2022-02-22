"""
// Copyright (c) 2019 Robert Bosch GmbH
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

#import glob
import os
import sys
import csv
import pickle
import argparse
import numpy as np
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from default_parameters import default_parameters_OWEO
from my_toolkit import NX_matrix_transformation, give_U_name, add_NX_structure, pt1_filter

def main(NX_matrix_inputs, data_dir, used_measurement, save_processed_data, pt1_inputs, pt1_outputs):
    X_name = ["Speed",
              "Load",
              "Lambda",
              "Ign_Ang",
              "Fuel_Cutoff"]
    
    Y_name = ["PAR",
              "CO",
              "CO2",
              "HC",
              "NOx",
              "O2",
              "T_EXM",
              "T_CAT1"]
    
    NX_matrix = NX_matrix_transformation(NX_matrix_inputs, len(X_name))
    U_name = give_U_name(X_name, NX_matrix)
    
    X = np.empty([0, len(X_name)])
    filt_X = np.empty([0, len(X_name)])
    U = np.empty([0, len(U_name)])
    filt_U = np.empty([0, len(U_name)])
    Y = np.empty([0, len(Y_name)])
    filt_Y = np.empty([0, len(Y_name)])
    
    for m_idx, m_value in enumerate(used_measurement):
        # first load data from a file
        data = np.empty([0,len(X_name)+len(Y_name)])
        file = os.path.join(data_dir, f"raw_0000{m_value}_measurement_000000.csv")
        print("### measurement %3d / %s"%(m_value, str(used_measurement[m_idx:])))
        
        with open(file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            print("###    loading", end='')
            
            for row in csvreader:
                data = np.vstack((data,row))
        
        print(", filtering", end='')
        filt_data = np.empty(np.shape(data))
        for i, pt1 in enumerate(np.append(pt1_inputs, pt1_outputs)):
            filt_data[:, i] = pt1_filter(data[:, i], pt1)
        
        print(", adding NX structure", end='')
        uu = add_NX_structure(data, NX_matrix)
        filt_uu = add_NX_structure(filt_data, NX_matrix)
        
        print(", finishing\n")
        steps = NX_matrix.shape[0] - 1
        
        X = np.vstack((X, data[steps:, :len(X_name)]))
        filt_X = np.vstack((filt_X, filt_data[steps:, :len(X_name)]))
        U = np.vstack((U, uu))
        filt_U = np.vstack((filt_U, filt_uu))
        Y = np.vstack((Y, data[steps:, len(X_name):]))
        filt_Y = np.vstack((filt_Y, filt_data[steps:, len(X_name):]))
    
    # save result
    Processed_data = {
            "X_name": X_name, "X_raw": X, "X": filt_X,
            "U_name": U_name, "U_raw": U, "U": filt_U,
            "Y_name": Y_name, "Y_raw": Y, "Y": filt_Y,
            "NX_matrix": NX_matrix, "pt1_inputs": pt1_inputs, "pt1_outputs": pt1_outputs
            }

    if save_processed_data[0]:
        with open(save_processed_data[1], "wb") as fp:
            pickle.dump(Processed_data, fp)
        
    return Processed_data




if __name__ == "__main__":
    pars = default_parameters_OWEO()
    parser = argparse.ArgumentParser(description='Add NX structure to the data')
    parser.add_argument('--mode', default= "training", type=str, help="'training' or 'test' data are we processing")
    parser.add_argument('--NX_matrix', default= pars.NX_pos_str, nargs='+', type=str, help="format 'rXcY' or None, X & Y are indices of row & column of NX_matrix that are set to 1 (0 <= Y <= 4), the other entries will be 0")
    parser.add_argument('--pt1_inputs', default= pars.pt1_X, nargs='+', type=int, help=f"default={pars.pt1_X}, pt1 of input channels, 1 if no filter")
    parser.add_argument('--pt1_outputs', default= pars.pt1_Y, nargs='+', type=int, help=f"default={pars.pt1_Y}, pt1 of output channels, 1 if no filter")
    parser.add_argument('--data_dir', default= pars.raw_data_dir, type=str, help=f"default={pars.raw_data_dir}, the folder where the input raw data are")
    parser.add_argument('--used_measurement_training', default= pars.used_measurement_training, nargs='+', type=int, help=f"default={pars.used_measurement_training}, indices of measurement we want use for training (default is just fine)")
    parser.add_argument('--used_measurement_test', default= pars.used_measurement_test, nargs='+', type=int, help=f"default={pars.used_measurement_test}, indices of measurement we want use for test (default is just fine)")
    parser.add_argument('--filename_training', default= pars.filename_training, type=str, help=f"default={pars.filename_training}, full path of processed training data")
    parser.add_argument('--filename_test', default= pars.filename_test, type=str, help=f"default={pars.filename_test}, full path of processed test data")
    parser.add_argument('--save_processed_data', default=True, type=bool, help="default=True, whether we save the processed data or not")
    
    args = parser.parse_args()
    
    mode = args.mode
    data_dir = args.data_dir
    
    used_measurement = args.used_measurement_training \
        if mode == "training" else args.used_measurement_test
    save_processed_data = [args.save_processed_data, args.filename_training] \
        if mode == "training" else [args.save_processed_data, args.filename_test]
    
    NX_matrix_inputs = args.NX_matrix
    pt1_inputs = np.array(args.pt1_inputs)
    pt1_outputs = np.array(args.pt1_outputs)
    
    Processed_data = main(NX_matrix_inputs, data_dir, used_measurement, save_processed_data, pt1_inputs, pt1_outputs)











