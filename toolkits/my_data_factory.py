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
import os
import sys

sys.path.append(os.path.dirname(__file__))

from my_toolkit import raw_wise_compare, add_series, data_pairs_generator,\
    data_selection, create_args, args_partially_queried_matrix,\
    extract_name_tuple, extract_data_tuple,\
    result_placeholder,\
    loadfile, savefile, save_model, save_model_dict

gpflow.config.set_default_summary_fmt("notebook")


class data_manager_toy():
    def __init__(
            self,
            exploration_interval,
            true_function,
            true_safety_function,
            noise_function,
            noise_std,
            noise_std_safety,
            num_init_data: int,
            input_dim: int=1,
            init_interval = [-0.8, 0.8],
            test_interval = [-1.2, 1.2],
            series_half_step_num:int=0,
            series_step:float=0.1,
            POO: bool = False
    ):
        self.noise_std = np.reshape(noise_std, [-1])
        self.noise_std_safety = np.reshape(noise_std_safety, [-1])

        self.shsn = series_half_step_num
        self.sstp = series_step

        self.ex_interval = exploration_interval
        self.test_interval = test_interval

        self.true_function = true_function
        self.true_safety_function = true_safety_function
        self.noise_function = noise_function
        
        self.input_dim = input_dim
        self.num_init_data = num_init_data
        self.init_interval = init_interval
        self.POO = POO

    def true_safe_training_tuple(self, Z_threshold, safe_above_threshold=True):
        x, u, _, _ = self.training_pool
        z = self.true_safety_function(x).reshape(-1)
        
        if safe_above_threshold:
            mask = (z >= Z_threshold)
        else:
            mask = (z <= Z_threshold)

        x_eval = x[mask, ...]
        u_eval = u[mask, ...]
        y_eval = self.true_function(x_eval)
        z_eval = z[mask, ...]
        
        return (x_eval, u_eval, y_eval, z_eval)


    def set_seed(self, seed=123):
        np.random.seed(seed)

    def add_series(self, X):
        return add_series(X, half_step_num=self.shsn, step=self.sstp)

    def remove_series(self, U):
        return U[:, self.shsn, None]

    def create_tuples(self):
        if self.input_dim == 1:
            X_tr = np.linspace(*self.ex_interval, num=200)[:,None]
        else:
            X_tr = np.random.uniform(*self.ex_interval, size=[200, self.input_dim])
        self.training_pool = self.create_sample(X=X_tr)
        self.test_pool = self.create_sample(interval=self.test_interval, num=3*self.num_init_data)
        self.name_tuple = (
            ['x'],
            [f'x-{i * self.sstp}' for i in range(-self.shsn, self.shsn + 1)],
            [f'y{i + 1}' for i in range(self.return_dimensions('Y'))],
            ['z']
        )

    def create_sample(self, interval=[0, 1], num=0, X=None):
        if X is None:
            X = np.random.uniform(*interval, size=[num, self.input_dim])
        else:
            num = np.shape(X)[0]
        U = self.add_series(X)
        Y = self.true_function(X)
        Y+= self.noise_function(Y, self.noise_std)
        Z = self.true_safety_function(X)
        Z+= self.noise_function(Z, self.noise_std_safety)

        if self.POO:
            P = Y.shape[1]
            mask = np.zeros([num, P], dtype=bool)
            for i in range(P):
                mask[i::P, i] = True
            
            Y = np.where(mask, Y, np.nan * np.ones_like(Y))

        return (X, U, Y, Z)

    def return_tuples(self ,key):
        if key.lower() == 'name_tuple':
            return self.name_tuple
        if key.lower() == 'training_pool':
            return self.training_pool
        if key.lower() == 'test_pool':
            return self.test_pool

    def return_dimensions(self, variable=None):
        X, U, Y, Z = self.training_pool
        if variable is None:
            return np.shape(X)[1], np.shape(U)[1], np.shape(Y)[1], np.shape(Z)[1]
        elif variable.lower() == 'x':
            return np.shape(X)[1]
        elif variable.lower() == 'u':
            return np.shape(U)[1]
        elif variable.lower() == 'y':
            return np.shape(Y)[1]
        elif variable.lower() == 'z':
            return np.shape(Z)[1]

    def pool_size(self, data='training'):
        if data.lower() == 'training':
            N = np.shape(self.training_pool[0])[0]
        elif data.lower() == 'test':
            N = np.shape(self.test_pool[0])[0]
        else:
            raise ValueError('unrecognized input')
        return N

    # the following works for data_exclusive models
    # for data_inclusive models, use model.updata_dataset()
    def return_training_data_tuple(self, key=None):
        if not key is None:
            return self.X[key], self.U[key], self.Y[key], self.Z[key]
        else:
            return self.X, self.U, self.Y, self.Z

    def training_data_selection(self, data_args):
        return data_selection(self.training_pool, data_args)

    def training_data_initializer(self, model_names, data_args):
        self.model_names = model_names
        self.X, self.U, self.Y, self.Z = {}, {}, {}, {}
        self.training_data_args = {}
        for j, name in enumerate(model_names):
            self.training_data_args[name] = data_args
            self.X[name], self.U[name], self.Y[name], self.Z[name] = data_selection(self.training_pool, data_args)

    def training_data_update(self, args, key=None):
        if key is None:
            for j, name in enumerate(self.model_names):
                X_new, U_new, Y_new, Z_new = self.training_data_selection(args[j ,...])
                self.X[name] = np.vstack((self.X[name], X_new))
                self.U[name] = np.vstack((self.U[name], U_new))
                self.Y[name] = np.vstack((self.Y[name], Y_new))
                self.Z[name] = np.vstack((self.Z[name], Z_new))
        else:
            X_new, U_new, Y_new, Z_new = self.training_data_selection(args)
            self.X[key] = np.vstack((self.X[key], X_new))
            self.U[key] = np.vstack((self.U[key], U_new))
            self.Y[key] = np.vstack((self.Y[key], Y_new))
            self.Z[key] = np.vstack((self.Z[key], Z_new))


class data_manager_GPsamples():
    def __init__(
            self,
            exp_idx,
            data_dir,
            noise_function,
            noise_std,
            noise_std_safety,
            num_init_data: int,
            POO: bool = False
    ):
        self.exp_idx = exp_idx
        self.data_dir = data_dir
        self.noise_std = np.reshape(noise_std, [-1])
        self.noise_std_safety = np.reshape(noise_std_safety, [-1])

        self.num_init_data = num_init_data
        self.noise_function = noise_function
        self.POO = POO

        self.training_raw_data = self.load_raw_data(dataset='training')
        self.test_raw_data = self.load_raw_data(dataset='test')
    
    def load_raw_data(self, dataset='training'):
        if dataset == 'training':
            X = loadfile(os.path.join(self.data_dir, 'X_training.pkl'), mode='rb')
            F = loadfile(os.path.join(self.data_dir, f'F_training_exp{self.exp_idx}.pkl'), mode='rb')
            Z = loadfile(os.path.join(self.data_dir, f'Z_training_exp{self.exp_idx}.pkl'), mode='rb')
        elif dataset == 'test':
            X = loadfile(os.path.join(self.data_dir, 'X_test.pkl'), mode='rb')
            F = loadfile(os.path.join(self.data_dir, f'F_test_exp{self.exp_idx}.pkl'), mode='rb')
            Z = loadfile(os.path.join(self.data_dir, f'Z_test_exp{self.exp_idx}.pkl'), mode='rb')
        return (X, F, Z)

    def true_safe_tuple(self, Z_threshold, safe_above_threshold=True, dataset='training'):
        if dataset == 'training':
            x, u, _, _ = self.training_pool
            _, y, z = self.training_raw_data
        elif dataset == 'test':
            x, u, _, _ = self.test_pool
            _, y, z = self.test_raw_data
        else:
            raise ValueError("The dataset must be str 'training' or 'test'.")

        if safe_above_threshold:
            mask = (z >= Z_threshold).reshape(-1)
        else:
            mask = (z <= Z_threshold).reshape(-1)

        x_eval = x[mask, ...]
        u_eval = u[mask, ...]
        y_eval = y[mask, ...]
        z_eval = z[mask, ...]
        
        return (x_eval, u_eval, y_eval, z_eval)

    def create_tuples(self):
        X, F, Z = self.training_raw_data
        self.training_pool = (
            X,
            X,
            F + self.noise_function(F, self.noise_std),
            Z + self.noise_function(Z, self.noise_std_safety)
        )
        
        X, F, Z = self.test_raw_data
        self.test_pool = (
            X,
            X,
            F + self.noise_function(F, self.noise_std),
            Z + self.noise_function(Z, self.noise_std_safety)
        )
        
        self.name_tuple = (
            [f'x{i + 1}' for i in range(self.return_dimensions('X'))],
            [f'x{i + 1}' for i in range(self.return_dimensions('U'))],
            [f'y{i + 1}' for i in range(self.return_dimensions('Y'))],
            ['z']
        )

    def create_sample(self, num=0, safe=True, Z_threshold=0):
        if safe:
            Xsafe, _, _, _ = self.true_safe_tuple(Z_threshold, dataset='training')
            X, U, Y, Z = self.training_pool
            idx = raw_wise_compare(X, Xsafe)
            X = X[idx,...]
            U = U[idx,...]
            Y = Y[idx,...]
            Z = Z[idx,...]
        else:
            X, U, Y, Z = self.training_pool
        
        idx = np.random.permutation(X.shape[0])[:num]

        X = X[idx, :].reshape([num, X.shape[1]])
        U = U[idx, :].reshape([num, U.shape[1]])
        Y = Y[idx, :].reshape([num, Y.shape[1]])
        Z = Z[idx, :].reshape([num, Z.shape[1]])
        
        if self.POO:
            P = Y.shape[1]
            mask = np.zeros([num, P], dtype=bool)
            for i in range(P):
                mask[i::P, i] = True
            
            Y = np.where(mask, Y, np.nan * np.ones_like(Y))

        return (X, U, Y, Z)

    def return_tuples(self ,key):
        if key.lower() == 'name_tuple':
            return self.name_tuple
        if key.lower() == 'training_pool':
            return self.training_pool
        if key.lower() == 'test_pool':
            return self.test_pool

    def return_dimensions(self, variable=None):
        X, U, Y, Z = self.training_pool
        if variable is None:
            return np.shape(X)[1], np.shape(U)[1], np.shape(Y)[1], np.shape(Z)[1]
        elif variable.lower() == 'x':
            return np.shape(X)[1]
        elif variable.lower() == 'u':
            return np.shape(U)[1]
        elif variable.lower() == 'y':
            return np.shape(Y)[1]
        elif variable.lower() == 'z':
            return np.shape(Z)[1]

    def pool_size(self, data='training'):
        if data.lower() == 'training':
            N = np.shape(self.training_pool[0])[0]
        elif data.lower() == 'test':
            N = np.shape(self.test_pool[0])[0]
        else:
            raise ValueError('unrecognized input')
        return N

    # the following works for data_exclusive models
    # for data_inclusive models, use model.updata_dataset()
    def return_training_data_tuple(self, key=None):
        if not key is None:
            return self.X[key], self.U[key], self.Y[key], self.Z[key]
        else:
            return self.X, self.U, self.Y, self.Z

    def training_data_selection(self, data_args):
        return data_selection(self.training_pool, data_args)

    def training_data_initializer(self, model_names, data_args):
        self.model_names = model_names
        self.X, self.U, self.Y, self.Z = {}, {}, {}, {}
        self.training_data_args = {}
        for j, name in enumerate(model_names):
            self.training_data_args[name] = data_args
            self.X[name], self.U[name], self.Y[name], self.Z[name] = data_selection(self.training_pool, data_args)

    def training_data_update(self, args, key=None):
        if key is None:
            for j, name in enumerate(self.model_names):
                X_new, U_new, Y_new, Z_new = self.training_data_selection(args[j ,...])
                self.X[name] = np.vstack((self.X[name], X_new))
                self.U[name] = np.vstack((self.U[name], U_new))
                self.Y[name] = np.vstack((self.Y[name], Y_new))
                self.Z[name] = np.vstack((self.Z[name], Z_new))
        else:
            X_new, U_new, Y_new, Z_new = self.training_data_selection(args)
            self.X[key] = np.vstack((self.X[key], X_new))
            self.U[key] = np.vstack((self.U[key], U_new))
            self.Y[key] = np.vstack((self.Y[key], Y_new))
            self.Z[key] = np.vstack((self.Z[key], Z_new))


class data_manager_OWEO():
    def __init__(self, raw_data_training, raw_data_test, POO:bool=False):
        self.NX_matrix = raw_data_training["NX_matrix"]
        self.raw_data_training = raw_data_training
        self.raw_data_test = raw_data_test
        self.excluded_args = np.array([]) # this is only used for AL on subset
        self.POO = POO

    def create_tuples(self ,used_y_ind, used_z_ind):
        self.used_y_ind = list(np.reshape(used_y_ind ,[-1]))
        self.used_z_ind = list(np.reshape(used_z_ind ,[-1]))
        self.name_tuple = extract_name_tuple(self.raw_data_training, self.used_y_ind, self.used_z_ind)
        self.training_pool = extract_data_tuple(self.raw_data_training, self.used_y_ind, self.used_z_ind)
        self.test_pool = extract_data_tuple(self.raw_data_test, self.used_y_ind, self.used_z_ind)

    def return_tuples(self ,key):
        if key.lower() == 'name_tuple':
            return self.name_tuple
        if key.lower() == 'training_pool':
            return self.training_pool
        if key.lower() == 'test_pool':
            return self.test_pool

    def return_dimensions(self, variable=None):
        X, U, Y, Z = self.training_pool
        if variable is None:
            return np.shape(X)[1], np.shape(U)[1], np.shape(Y)[1], np.shape(Z)[1]
        elif variable.lower() == 'x':
            return np.shape(X)[1]
        elif variable.lower() == 'u':
            return np.shape(U)[1]
        elif variable.lower() == 'y':
            return np.shape(Y)[1]
        elif variable.lower() == 'z':
            return np.shape(Z)[1]

    def pool_size(self, data='training'):
        if data.lower() == 'training':
            N = np.shape(self.raw_data_training['U'])[0]
        elif data.lower() == 'test':
            N = np.shape(self.raw_data_test['U'])[0]
        else:
            raise ValueError('unrecognized input')
        return N

    def generate_training_args(
            self,
            input_dir: str,
            num_of_data: int,
            set_idx: int,
            safety_threshold: float=None,
            safe_above_threshold: bool=False,
            num_of_sets: int =100,
            num_preselect: int=None
    ):
        num = num_of_data if num_preselect is None else num_preselect
        """
        if safety_threshold is None:
            args_filename = os.path.join(input_dir, 'training_args_nosafe', 'args_' + str(num) +'.pkl')
        else:
            args_filename = os.path.join(input_dir, 'training_args', 'args_' + str(num) +'.pkl')
        if os.path.exists(args_filename):
            args_list = loadfile(args_filename, mode='rb')
        else:
            _, _, _, Z = self.training_pool
            N = Z.shape[0]
            args_pool = np.arange(N)
            if safety_threshold is None:
                safe_mask = np.ones(N, dtype=bool)
            elif safe_above_threshold:
                safe_mask = Z >= safety_threshold
                safe_mask = safe_mask.reshape(-1)
            else:
                safe_mask = Z <= safety_threshold
                safe_mask = safe_mask.reshape(-1)
            args_list = create_args(args_filename, args_pool[safe_mask], num, setnum=num_of_sets)
        """
        _, _, _, Z = self.training_pool
        N = Z.shape[0]
        args_pool = np.arange(N)
        if safety_threshold is None:
            safe_mask = np.ones(N, dtype=bool)
        elif safe_above_threshold:
            safe_mask = Z >= safety_threshold
            safe_mask = safe_mask.reshape(-1)
        else:
            safe_mask = Z <= safety_threshold
            safe_mask = safe_mask.reshape(-1)
        args_list = create_args(None, args_pool[safe_mask], num, setnum=num_of_sets, saveargs=False)

        args = args_list[set_idx ,:]
        
        if not num_preselect is None:
            self.training_data_preselection(args)
            args = args[:num_of_data]

        if self.POO:
            return self.add_task_index(args)
        else:
            return args


    def training_data_preselection(self, data_args): # this is only used for AL on subset
        N = self.pool_size('training')
        data_args = np.array(data_args, dtype=int)
        self.excluded_args = np.delete(np.arange(N, dtype=int), data_args)

    def add_task_index(self, data_args:np.ndarray):
        P = self.return_dimensions('Y')
        
        mask = np.reshape(
            np.arange(len(data_args.reshape(-1))) % P,
            [1,-1]
        )
        
        return np.vstack((
            np.reshape(data_args, [1, -1]),
            mask
        ))

    # the following works for data_exclusive models
    # for data_inclusive models, use model.updata_dataset()
    def return_training_data_tuple(self, key=None):
        if not key is None:
            return self.X[key], self.U[key], self.Y[key], self.Z[key]
        else:
            return self.X, self.U, self.Y, self.Z

    def training_data_selection(self, data_args):
        return data_selection(self.training_pool, data_args)

    def training_data_initializer(self, model_names, data_args):
        self.model_names = model_names
        self.X, self.U, self.Y, self.Z = {}, {}, {}, {}
        self.training_data_args = {}
        for j, name in enumerate(model_names):
            self.training_data_args[name] = data_args
            self.X[name], self.U[name], self.Y[name], self.Z[name] = data_selection(self.training_pool, data_args)

    def training_data_update(self, args, key=None):
        if key is None:
            for j, name in enumerate(self.model_names):
                X_new, U_new, Y_new, Z_new = self.training_data_selection(args[j ,...])
                self.X[name] = np.vstack((self.X[name], X_new))
                self.U[name] = np.vstack((self.U[name], U_new))
                self.Y[name] = np.vstack((self.Y[name], Y_new))
                self.Z[name] = np.vstack((self.Z[name], Z_new))
        else:
            X_new, U_new, Y_new, Z_new = self.training_data_selection(args)
            self.X[key] = np.vstack((self.X[key], X_new))
            self.U[key] = np.vstack((self.U[key], U_new))
            self.Y[key] = np.vstack((self.Y[key], Y_new))
            self.Z[key] = np.vstack((self.Z[key], Z_new))