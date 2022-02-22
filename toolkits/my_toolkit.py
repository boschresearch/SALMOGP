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
import pickle
import glob
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from distutils import util

def str2bool(v):
    return bool(util.strtobool(str(v)))


def raw_wise_compare(x, y):
    r"""

    :param x: [N1, D] array
    :param y: [N2, D] array
    :return: [N1, ] boolean array, True if the row in x is contained in y
    """
    x = np.array(x)
    y = np.array(y)

    d1 = x.shape[1]
    d2 = y.shape[1]

    struct_x = x.view(x.dtype.descr * d1)
    struct_y = y.view(y.dtype.descr * d2)

    return np.in1d(struct_x, struct_y)


def return_unused_args(data, pool):
    r"""
    data: (X, Y)
    pool: (Xp, Yp)

    return args of points which are in pool but not in data
    [[i, ...],
     [p, ...]]
        i are indices in Xp
        p are output channel indices
    Y might be partially observed
    """
    X, Y = data
    Xp, Yp = pool

    mask = np.zeros(Yp.shape, dtype=bool)
    for p in range(Yp.shape[1]):
        mask[:,p] = raw_wise_compare(Xp, X[~np.isnan(Y[:,p])])
    
    return np.vstack(np.where(~mask))
    
### Toy data
#######################################################################
def sigmoid_function(x, scale=1.0):
    return 1/(1 + np.exp(-x/scale))


def add_series(x, half_step_num:int=2, step:float=0.1):
    """
    :param x: input data, [N, 1]
    :param half_step_num: 
    :param step: 
    :return: [x - half_step_num * step, x - (half_step_num-1)* step, ..., x, x+step, ..., x + half_step_num * step]
    """
    if half_step_num == 0:
        return x
    N = np.shape(x)[0]
    x_series = np.zeros([N, 1 + 2*half_step_num])
    x_series[:, half_step_num] = x.reshape(-1)
    for i in range(1, half_step_num+1):
        x_series[:, half_step_num - i] = x.reshape(-1) - step
        x_series[:, half_step_num + i] = x.reshape(-1) + step

    return x_series


def real_function(X, mode=None):
    Y = np.hstack((
        np.sin(10*X) + sigmoid_function(X),
        np.sin(10*X) - sigmoid_function(X)#+ np.cos(10*X)
        ))
    return Y


def noise_function(X, std):
    # Y=aX, X~ N(0,1), then Y~N(0, a^2)
    nn = np.random.randn(X.shape[0], len(std))
    
    nn *= std
    
    return  nn


def XY_map(X, std):
    return real_function(X) + noise_function(X, std)


def data_pairs_generator(std, num_init_data, mode="training"):
    # make a dataset with two outputs, correlated, heavy-tail noise. One has more noise than the other.
    if mode == "training":
        # X = np.random.rand(num_init_data, 1)
        X = np.random.uniform(-1.0, 1.0, size=[num_init_data, 1])
    elif mode == "test":
        X = np.random.uniform(-1.2, 1.2, size=[num_init_data, 1])
    
    Y = XY_map(X, std)
    return X, Y


def safety_function_noise_free(X_augmented):
    if len(np.shape(X_augmented)) < 2:
        Z = np.power(X_augmented - 0.1, 2) / 2
    else:
        Z = np.power(X_augmented[:,0]  - 0.1, 2) / 2

    return np.exp(-Z).reshape([-1,1])


def safety_function(X_augmented, std):
    Z = safety_function_noise_free(X_augmented)
    
    nn = np.random.randn(*Z.shape)
    nn *= std
    
    return Z+nn


### functions for OWEO pipeline
#######################################################################

def args_partially_queried_matrix(args, num_task:int):
    r"""
    input: args: [2, N] matrix
                 args[0,:] are args of data
                 args[1,:] are used task idx
    
    output: pd.DataFrame, [k, num_task] bool matrix
        with columns = task_idx (e.g. [0, 1, 2, ...])
             index = data_args in args[0,:]
    notice: output[task_idx][arg] = True if [[args], [task_idx]] is in args, otherwise False
    """
    args_base, inv_idx = np.unique(args[0,:], return_inverse=True)
    queried_mtx = np.zeros([len(args_base), num_task], dtype=bool)
    queried_mtx[inv_idx, args[1,:]] = True
    
    return pd.DataFrame(queried_mtx, index = args_base)
    

def data_selection(data_pool, arg):
    r"""
    input data tuple (X, U, Y, Z), and args
    pick the corresponding samples from the pool
    
    output: X[args], U[args], Y[args], Z[args]
    """
    X, U, Y, Z = data_pool
    if len(np.shape(arg)) == 1 or (len(np.shape(arg)) == 2 and np.shape(arg)[0] == 1):
        return X[arg,:].reshape([-1, X.shape[1]]), U[arg,:].reshape([-1, U.shape[1]]), Y[arg,:].reshape([-1, Y.shape[1]]), Z[arg,:].reshape([-1, Z.shape[1]])
    elif len(np.shape(arg)) == 2:
        mask = np.zeros(Y.shape, dtype=bool)
        mask[arg[0,:], arg[1,:]] = True
        
        valid_raw = (mask.sum(axis=1) > 0)

        Xs = X[valid_raw]
        Us = U[valid_raw]
        Ys = Y[valid_raw]
        Zs = Z[valid_raw]

        Ys = np.where(mask[valid_raw], Ys, np.nan * np.ones_like(Ys))

        return Xs, Us, Ys, Zs


def extract_name_tuple(raw_data, used_y_ind, used_z_ind):
    X_name = raw_data["X_name"]
    U_name = raw_data["U_name"]
    Y_name = [raw_data["Y_name"][ind] for ind in used_y_ind]
    Z_name = [raw_data["Y_name"][ind] for ind in used_z_ind]
    name_tuple = (X_name, U_name, Y_name, Z_name)
    return name_tuple
    

def extract_data_tuple(raw_data, used_y_ind, used_z_ind):
    X = raw_data["X"]
    U = raw_data["U"]
    Y = raw_data["Y"][:, used_y_ind]
    Z = raw_data["Y"][:, used_z_ind]
    
    if len(used_y_ind)==1:
        Y = Y.reshape([-1,1])
    if len(used_z_ind)==1:
        Z = Z.reshape([-1,1])
    
    data_tuple = (X, U, Y, Z)
    return data_tuple


def result_placeholder(labels, shape=[1]):
    
    placeholder = {
            labels[ind]: np.zeros(shape) for ind in range(len(labels))
            }
    
    return placeholder


def loadfile(fullpath, mode='rb'):
    with open(fullpath, mode) as fp:
        data = pickle.load(fp)
    return data


def savefile(fullpath, data, mode='wb'):
    with open(fullpath, mode) as fp:
        pickle.dump(data, fp)
    return True


def save_model(fullpath, model, mode='w', save_ivs=True, save_variational_pars=True, save_history=True):
    from gpflow.utilities import read_values
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    model_pars = read_values(model)
    model_hist = model.history
    with open(fullpath, mode) as fp:
        print('model parameters\n----------------', file=fp)
        for name, content in model_pars.items():
            if not save_ivs and name[:18] == '.inducing_variable':
                continue
            if not save_variational_pars and name[:5] == '.q_mu':
                continue
            if not save_variational_pars and name[:7] == '.q_sqrt':
                continue
            
            if len(np.shape(content)) <= 1:
                print(f'{name}: {content}', file=fp)
            else:
                print(f'{name}:\n{content}', file=fp)
        
        if save_history:
            print('\nmodel history\n-------------', file=fp)
            for name, content in model_hist.items():
                if len(np.shape(content)) <= 1:
                    print(f'{name}: {content}', file=fp)
                else:
                    print(f'{name}:\n{content}', file=fp)
    return True


def save_model_dict(fullpath, model_dict, mode='w', save_ivs=True, save_variational_pars=True):
    for name, model in model_dict.items():
        save_model(fullpath+str(name)+'.txt', model, mode, save_ivs, save_variational_pars)
    return True


def create_args(fullpath, args_pool, argsnum, setnum=100, seed=123, saveargs=True):
    np.random.seed(seed)
    
    args_list = np.empty([setnum, argsnum], dtype=int)
    for nn in range(setnum):
        args_list[nn, :] = np.random.choice(args_pool, argsnum, replace=False)
        
    if saveargs:
        argsdir, _ = os.path.split(fullpath)
        Path(argsdir).mkdir(parents=True, exist_ok=True)
        savefile(fullpath, args_list, mode='wb')
    
    return args_list


### functions for OWEO pipeline, data processing
#######################################################################

def NX_matrix_transformation(matrix_strings, max_col = 5):
    r"""
    input list = [..., 'rXcY', ...], where X, Y are int
    return binary matrix with matrix[X, Y]=1, 0 elsewhere
    """
    if matrix_strings is None or matrix_strings is [None]:
        return np.ones([1, max_col], dtype=int)
    
    row = []
    col = []
    for string in matrix_strings:
        _, string = string.split('r')
        ind1, ind2 = string.split('c')
        
        row.append(int(ind1))
        col.append(int(ind2))
    
    NX_matrix = np.zeros([max(row) + 1, max_col], dtype=int)
    
    for i in range(len(row)):
        NX_matrix[row[i], col[i]] = 1
    
    return NX_matrix


def give_U_name(X_name, NX_matrix):
    U_name = []
    
    for i in range(len(X_name)):
        t = -np.arange(NX_matrix.shape[0])[NX_matrix[:, i]==1]
        for tt in t:
            if tt == 0:
                U_name.append(X_name[i]+", t")
            else:
                U_name.append(X_name[i]+", t"+str(tt))
    
    return U_name


def add_NX_structure(data, NX_matrix):
    r"""
    input
    data: [N, D] array
    NX_matrix: [T, D] array, element = 0 or 1
    
    output: [N, P] array, P = sum(NX_matrix)
    """
    steps = np.shape(NX_matrix)[0] - 1
    U = np.empty([data.shape[0]-steps, 0])
    for j in range(NX_matrix.shape[1]):
        t = -np.arange(steps+1)[NX_matrix[:, j]==1]
        for tt in t:
            if tt==0:
                U = np.hstack((U, data[steps:, j, None]))
            else:
                U = np.hstack((U, data[steps+tt:tt, j, None]))
            
    return U


def pt1_filter(x, pt1, axis=-1):
    r"""
    this is a filter with b = [1/pt1], a = [1, 1/pt1 - 1]
    a[0]y[n] = b[0]x[n] - a[1]y[n-1]
    
    see scipy.signal.lfilter for more info
    """
    b = np.array([1/pt1])
    a = np.array([1, 1/pt1 - 1])
    output = signal.lfilter(b, a, x, axis=axis)
    return output


### functions experimental result collection
#######################################################################
def collect_results(file_list):
    r"""
    file_list cannot be empty,
    all files in the list must have exactly the same structure
    """
    with open(file_list[0], "rb") as fp:
        dict_tmp = pickle.load(fp)
        collection = {
                name: np.zeros([len(file_list), *np.shape(content)])
                    for name, content in dict_tmp.items()
                }
    for i, file in enumerate(file_list):
        # load result of the i-th trial
        result_per_trial = loadfile(file, mode='rb')
        # put the result into the corresponding place in collection
        for name, content in result_per_trial.items():
            if name == 'X' or name == 'True_Y':
                continue
            collection[name][i,...] = content
    
    return collection


def collection_filename(full_filename_one_trial, dir_jump_level=1, split_str='_exp', extension = '.pkl'):
    r"""
    full_filename_one_trial = '/a0/a1/.../an/' + str0 + split_str + str1
    dir_jump_level = k
    
    return: '/a0/.../a(n-k)/' + str0 + extension
    """
    dir_name, basename = os.path.split(full_filename_one_trial)
    
    for _ in range(dir_jump_level):
        dir_name, _ = os.path.split(dir_name)
        
    basename, _ = basename.split(split_str)
    
    if dir_jump_level < 0:
        return basename + extension
    return os.path.join(dir_name, basename + extension)


def collect_all(file_dir, text, exp_idx_return=False, exp_idx=None):
    if exp_idx is None:
        files = glob.glob(os.path.join(file_dir, "individual_trials", "*" + text + "_exp*.pkl"))
        if exp_idx_return:
            exp_idx = []
            for name in files:
                _, name = name.split('_exp')
                idx, _ = name.split('.pkl')
                exp_idx.append(int(idx))
    else:
        files = []
        for idx in exp_idx:
            files.append(os.path.join(
                    file_dir,
                    "individual_trials",
                    text + f"_exp{idx}.pkl"
                    ))
    data = collect_results(files)
    savefile(collection_filename(files[0], 1, '_exp', ".pkl"), data, mode='wb')
    
    if exp_idx_return:
        return data, exp_idx
    return data






