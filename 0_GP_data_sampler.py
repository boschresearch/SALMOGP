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
import sys
import csv
import pickle
import argparse
import numpy as np
import tensorflow as tf
import gpflow
import datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(".","toolkits"))
from default_parameters import default_parameters_GPsamples
from pathlib import Path
from my_toolkit import str2bool, loadfile, savefile
import matplotlib.pyplot as plt

tz = datetime.datetime.now().astimezone().tzinfo

if __name__ == "__main__":
    pars = default_parameters_GPsamples()
    parser = argparse.ArgumentParser(description='Generate (X, Y) pairs with linear model of coregionalization')
    
    parser.add_argument('--seed', default= 123, type=int, help=f"default=123, seed we use to generate samples")
    parser.add_argument('--num_datasets', default= 30, type=int, help=f"default=30, number of training datasets and test datasets")
    parser.add_argument('--N_training', default= 2000, type=int, help=f"default=2000, size of training dataset")
    parser.add_argument('--N_test', default= 500, type=int, help=f"default=500, size of test dataset")
    parser.add_argument('--input_dim', default= pars.input_dim, type=int, help=f"default={pars.input_dim}, dimension of input data")
    parser.add_argument('--latent_dim', default= pars.latent_dim, type=int, help=f"default={pars.latent_dim}, number of latent GPs")
    parser.add_argument('--output_dim', default= pars.output_dim, type=int, help=f"default={pars.output_dim}, dimension of output data")
    parser.add_argument('--data_dir', default= pars.input_dir, type=str, help=f"default={pars.input_dir}, the folder where the input raw data are")
    
    args = parser.parse_args()

    datadir = args.data_dir
    Path(datadir).mkdir(parents=True, exist_ok=True)
    
    np.random.seed(args.seed)
    seed = np.random.get_state()[1][0]

    # set up parameters
    E = args.num_datasets
    N = args.N_training + args.N_test
    
    training_mask = np.zeros(N, dtype=bool)
    training_mask[:args.N_training] = True

    D = args.input_dim
    L = args.latent_dim
    P = args.output_dim

    # start generating data
    X_interval = [-2, 2]
    
    print('sampling X')
    X = np.random.uniform(*X_interval, size=[N, D]) # initialize
    while True:
        # make sure all raws are unique
        # this process also sort the first column
        _, idx = np.unique(X.view(X.dtype.descr * D), return_index=True)

        # permutation is required because np.unique sort the data
        # so otherwise the data won't really be random anymore
        X = X[np.random.permutation(idx)].reshape([-1, D])
        if X.shape[0] == N:
            break
        # this loop should end soon when points are not too dense in the space
        # if redundant raws excluded, add new ones
        X = np.vstack((
            X, np.random.uniform(*X_interval, size=[N - X.shape[0], D])
        ))

    # now samples noise free Y and Z
    # setup the GP kernels
    print('preparing GP kernels')
    W = np.random.multivariate_normal(np.zeros(L), np.eye(L), size=P)
    while True:
        # normalize, then we are sampling uniformly on a ball surface
        W = W[np.linalg.norm(W, axis=1, ord=2) > 0] # reject sample [0, 0, ..., 0]
        if W.shape[0] == P:
            break
        W = np.vstack((
            W, np.random.multivariate_normal(np.zeros(L), np.eye(L), size=(P - W.shape[0]))
        ))
    W = W / np.linalg.norm(W, axis=1, ord=2)[:,None]
    
    kernel_hypers = np.random.uniform(0.01, 1, size = [L+1, 2])
    kernel_hypers[:L,0] = 1.0 # variance of F absorted in W
    kernel_hypers[L,1] = 1.0 # lengthscale of safety function

    safe_kern = gpflow.kernels.SquaredExponential(
        variance=kernel_hypers[L,0],
        lengthscales=kernel_hypers[L,1]
        )
    kerns = []
    for l in range(L):
        kerns.append(
            gpflow.kernels.SquaredExponential(
                variance=kernel_hypers[l,0],
                lengthscales=kernel_hypers[l,1]
                )
        )
    
    K_list = [kern.K(X, X) for kern in kerns]

    # sampling
    print('sampling noise free Y, Z')
    latent_samples = np.concatenate([
        np.random.multivariate_normal(
            np.zeros(N),
            K,
            size=E
        )[:, :, None] for K in K_list
    ], axis=-1)

    F = latent_samples @ W.T
    Z = np.random.multivariate_normal(np.zeros(N), safe_kern.K(X, X), size=E)[:, :, None]

    print(f"X:{X.shape}\nG:{latent_samples.shape}\nF:{F.shape}\nZ:{Z.shape}")
    print(f"Z quantile: 20% : {np.quantile(Z, 0.2)}"+\
        f"\n            30% : {np.quantile(Z, 0.3)}"+\
        f"\n            40% : {np.quantile(Z, 0.4)}"+\
        f"\n            50% : {np.quantile(Z, 0.5)}"+\
        f"\n            60% : {np.quantile(Z, 0.6)}"+\
        f"\n            70% : {np.quantile(Z, 0.7)}"+\
        f"\n            80% : {np.quantile(Z, 0.8)}")
    # save samples
    savefile(os.path.join(datadir, 'X_training.pkl'), X[training_mask], mode='wb')
    savefile(os.path.join(datadir, 'X_test.pkl'), X[~training_mask], mode='wb')
    for e in range(E):
        savefile(os.path.join(datadir, f'F_training_exp{e}.pkl'), F[e,training_mask,...], mode='wb')
        savefile(os.path.join(datadir, f'F_test_exp{e}.pkl'), F[e,~training_mask,...], mode='wb')
        
        savefile(os.path.join(datadir, f'Z_training_exp{e}.pkl'), Z[e,training_mask,...], mode='wb')
        savefile(os.path.join(datadir, f'Z_test_exp{e}.pkl'), Z[e,~training_mask,...], mode='wb')


    with open(os.path.join(datadir, "data_parameters.txt"), mode="w") as fp:
        time_now = datetime.datetime.now(tz = tz)
        print(time_now.strftime('%Z (UTC%z)\n%Y.%b.%d  %A  %H:%M:%S\n'), file = fp)
        print("seed: %d \n"%(seed), file=fp)    
        print(f"X: np.random.uniform({X_interval[0]}, {X_interval[1]}, size=[{E}, {N}, {D}])", file=fp)
        print(f"   N_training : {args.N_training}", file=fp)
        print(f"   N_test     : {args.N_test}", file=fp)
        print(f"Z ~ GP(0, SE(variance={kernel_hypers[L,0]}, lengthscale={kernel_hypers[L,1]}))", file=fp)
        print(f"\nG: [L,1] vector", file=fp)
        for l in range(L):
            print(f"   G[{l}] ~ GP(0, SE(variance={kernel_hypers[l,0]}, lengthscale={kernel_hypers[l,1]}))", file=fp)
        print(f"\nF = W G", file=fp)
        print(f"   W = \n{W}", file=fp)
        print(f"\n\n\n\nZ quantile: 20% : {np.quantile(Z, 0.2)}"+\
              f"\n            30% : {np.quantile(Z, 0.3)}"+\
              f"\n            40% : {np.quantile(Z, 0.4)}"+\
              f"\n            50% : {np.quantile(Z, 0.5)}"+\
              f"\n            60% : {np.quantile(Z, 0.6)}"+\
              f"\n            70% : {np.quantile(Z, 0.7)}"+\
              f"\n            80% : {np.quantile(Z, 0.8)}", file=fp)
        
    fig, axs = plt.subplots(nrows = max(D, P+2), ncols=2)
    for i in range(D):
        axs[i, 0].plot(np.arange(N).reshape([-1,1]), X[:, i].T)
    for j in range(P):
        axs[j, 1].plot(np.arange(N).reshape([-1,1]), F[:, :, j].T)
    axs[P+1, 1].plot(np.arange(N).reshape([-1,1]), Z[:, :, 0].T)
    plt.show()