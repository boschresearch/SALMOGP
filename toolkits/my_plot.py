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
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(__file__))
from my_toolkit import data_selection

FIGSIZE_ROW = 6
FIGSIZE_COL = 8

def plot_data(X, Y, savefile=[False, "test.jpg"], display_fig=False):
    fig = plt.figure(figsize=(15, 8))
    for i in range(Y.shape[1]):
        plt.plot(X, Y[:,i], "x", mew=2, label="Y1")
    plt.legend()
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    return fig


def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )


def plot_model(m, training_data, test_data,
               real_function, safety_X,
               savefile=[False, "test.jpg"], 
               region=[-3.0, 3.0], ylim={"model":[-2, 2],
                                         "entropy":[-9, 3],
                                         "log_density": [-3, 3]},
               plot_iv=True, display_fig=False):
    # prepare all the arrays to be plotted
    Xcurve = np.linspace(*region, round(100*(region[1] - region[0])) )[:, None]
    X, Y = training_data
    X_test, Y_test = test_data
    history = m.history
    
    entropy = m.entropy(Xcurve, full_output_cov=True).numpy().reshape(-1)
    entropy_individuals = m.entropy(Xcurve, full_output_cov=False).numpy()
    
    mu, var = m.predict_f(Xcurve)
    mu, var = mu.numpy(), var.numpy()
    ground_truth = real_function(Xcurve)
    
    log_density = m.predict_log_density_full_output(test_data).numpy()
    
    
    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(15,8))
    fig.suptitle(f'variance: {m.likelihood.variance.numpy()}')
    for i in range(Y.shape[1]):
        # first plot the model and training data
        plt.sca(axs[0, 0])
        (line,) = plt.plot(X, Y[:, i], "x", mew=2)
        plot_gp(Xcurve, mu[:,i, None], var[:,i, None], line.get_color(), f"Y{i+1}")
        plt.plot(Xcurve, ground_truth[:, i], "--", color=line.get_color(), linewidth=2, label=f"Ground Truth {i+1}")
        
        # plot the model and test data
        plt.sca(axs[0, 1])
        plt.plot(X_test, Y_test[:, i],"x", color=line.get_color(), mew=2)
        plot_gp(Xcurve, mu[:,i, None], var[:,i, None], line.get_color(), f"Y{i+1}")
        plt.plot(Xcurve, ground_truth[:, i], "--", color=line.get_color(), linewidth=2, label=f"Ground Truth {i+1}")
        
        # plot entropy of individual tasks
        plt.sca(axs[1, 0])
        plt.plot(Xcurve, entropy_individuals[:, i], line.get_color())
        
        # plot test data & their log_dens
        plt.sca(axs[1, 1])
        plt.plot(X_test.reshape(-1), log_density[:, i], "o", color=line.get_color(), mew=0.8)
    
    plt.sca(axs[0, 0])
    plt.fill_between(
        safety_X, [-10000, -10000], [10000, 10000], color="g", alpha=0.4,label="safe region"
    )
    if plot_iv:
        iv = m.inducing_variable.inducing_variable.Z.numpy().reshape(-1)
        iv_lines = plt.plot(
                np.tile(iv, (2,1)), 
                np.vstack(([10000]*len(iv), [-10000]*len(iv))),
                "m--", linewidth=1.5)
        plt.setp(iv_lines[0], label="inducing locations")
    plt.legend()
    plt.ylim(ylim["model"])
    plt.title("Training time: %0.4f seconds"%(history["training_time"][-1]))
    
    plt.sca(axs[0, 1])
    plt.plot(np.tile(X_test.reshape(-1), (2,1)), 
                 np.vstack((np.max(Y_test, axis=1), [-10000]*Y_test.shape[0])),
                 "k--", linewidth=0.8)
    plt.legend()
    plt.ylim(ylim["model"])
    plt.title("test data")
    xlim = plt.gca().get_xlim()
    

    # entropy
    plt.sca(axs[1, 0])
    plt.plot(Xcurve, entropy, "k-", lw=2, label="differential entropy")
    plt.fill_between(
        safety_X, [-10000, -10000], [10000, 10000], color="g", alpha=0.4,
    )
    if plot_iv:
        iv_lines = plt.plot(
                np.tile(iv, (2,1)), 
                np.vstack(([10000]*len(iv), [-10000]*len(iv))),
                "m--", linewidth=1.5)
        plt.setp(iv_lines[0], label="inducing locations")
    
    plt.legend()
    plt.ylim(ylim["entropy"])
    title = "Entropy (multivariate)"
    if len(history["point_selection_time"]) > 0:
        title = title + ", data selection time: %0.4f seconds"%history["point_selection_time"][-1] 
    plt.title(title)
    
    # plot log density of test data
    plt.sca(axs[1, 1])
    plt.plot(np.tile(X_test.reshape(-1), (2,1)), 
             np.vstack((np.min(log_density, axis=1), [10000]*log_density.shape[0])),
             "k--", linewidth=0.8)
    
    plt.xlim(xlim)
    plt.ylim(ylim["log_density"])
    plt.ylabel("log density")
    #plt.title("RMSE:")
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    
    return fig


def plot_model_series(
        m, data_manager, safety_X,
        savefile=[False, "test.jpg"],
        region=[-3.0, 3.0], ylim={"model": [-2, 2],
                                  "entropy": [-9, 3],
                                  "log_density": [-3, 3]},
        display_fig=False):
    # prepare all the arrays to be plotted
    U, Y = m.data
    X = U[:, U.shape[1]//2, None]

    X_explor, U_explor, _, _ = data_manager.training_pool
    args = np.argsort(X_explor[:, 0])
    X_explor = X_explor[args].reshape([-1, 1])
    U_explor = U_explor[args].reshape([-1, data_manager.return_dimensions('U')])

    X_test, U_test, Y_test, Z_test = data_manager.test_pool
    history = m.history

    entropy = m.entropy(U_explor, full_output_cov=True).numpy().reshape(-1)
    entropy_individuals = m.entropy(U_explor, full_output_cov=False).numpy()

    mu, var = m.predict_f(U_explor)
    mu, var = mu.numpy(), var.numpy()
    ground_truth = data_manager.true_function(X_explor, mode='true')

    log_density = m.predict_log_density_full_output( (U_test, Y_test) ).numpy()

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(15, 8))
    fig.suptitle(f'variance: {m.likelihood.variance.numpy()}')
    for i in range(Y.shape[1]):
        # first plot the model and training data
        plt.sca(axs[0, 0])
        (line,) = plt.plot(X, Y[:, i], "x", mew=2)
        plot_gp(X_explor, mu[:, i, None], var[:, i, None], line.get_color(), f"Y{i + 1}")
        plt.plot(X_explor, ground_truth[:, i], "--", color=line.get_color(), linewidth=2, label=f"Ground Truth {i + 1}")

        # plot the model and test data
        plt.sca(axs[0, 1])
        plt.plot(X_test, Y_test[:, i], "x", color=line.get_color(), mew=2)
        plot_gp(X_explor, mu[:, i, None], var[:, i, None], line.get_color(), f"Y{i + 1}")
        plt.plot(X_explor, ground_truth[:, i], "--", color=line.get_color(), linewidth=2, label=f"Ground Truth {i + 1}")

        # plot entropy of individual tasks
        plt.sca(axs[1, 0])
        plt.plot(X_explor, entropy_individuals[:, i], line.get_color())

        # plot test data & their log_dens
        plt.sca(axs[1, 1])
        plt.plot(X_test.reshape(-1), log_density[:, i], "o", color=line.get_color(), mew=0.8)

    plt.sca(axs[0, 0])
    plt.fill_between(
        safety_X, [-10000, -10000], [10000, 10000], color="g", alpha=0.4, label="safe region"
    )

    plt.legend()
    plt.ylim(ylim["model"])
    plt.title("Training time: %0.4f seconds" % (history["training_time"][-1]))

    plt.sca(axs[0, 1])
    plt.plot(np.tile(X_test.reshape(-1), (2, 1)),
             np.vstack((np.max(Y_test, axis=1), [-10000] * Y_test.shape[0])),
             "k--", linewidth=0.8)
    plt.legend()
    plt.ylim(ylim["model"])
    plt.title("test data")
    xlim = plt.gca().get_xlim()

    # entropy
    plt.sca(axs[1, 0])
    plt.plot(X_explor, entropy, "k-", lw=2, label="differential entropy")
    plt.fill_between(
        safety_X, [-10000, -10000], [10000, 10000], color="g", alpha=0.4,
    )

    plt.legend()
    plt.ylim(ylim["entropy"])
    title = "Entropy (multivariate)"
    if len(history["point_selection_time"]) > 0:
        title = title + ", data selection time: %0.4f seconds" % history["point_selection_time"][-1]
    plt.title(title)

    # plot log density of test data
    plt.sca(axs[1, 1])
    plt.plot(np.tile(X_test.reshape(-1), (2, 1)),
             np.vstack((np.min(log_density, axis=1), [10000] * log_density.shape[0])),
             "k--", linewidth=0.8)

    plt.xlim(xlim)
    plt.ylim(ylim["log_density"])
    plt.ylabel("log density")
    # plt.title("RMSE:")

    if savefile[0]:
        plt.savefig(savefile[1])

    if not display_fig:
        plt.close(fig)

    return fig


def plot_safety_label(
        m, X_explor, training_data=None,
        savefile=[False, "test.jpg"],
        plot_iv=True, display_fig=False
        ):
    
    fig = plt.figure(figsize=(8,4))
    x = np.sort(X_explor[:, 0]).reshape([-1,1])
    mu, var = m.predict_f(x)
    if training_data is None:
        training_x, training_y = m.data
    else:
        training_x, training_y = training_data
    
    if plot_iv:
        iv = m.inducing_variable.inducing_variable.Z.numpy().reshape(-1)
        plt.plot(
            np.tile(iv, (2,1)), 
            np.vstack(([10000]*len(iv), [-10000]*len(iv))),
            "m--", linewidth=1.5)
    
    plot_gp(x, mu.numpy(), var.numpy(), "k", "safety GP model")
    plt.plot(training_x, training_y, "kx", label="data")
    plt.plot([-1, 1.5], [m.Z_threshold, m.Z_threshold], "g--", label="safety_threshold")
    
    plt.title("Safety labels and GP predictions")
    plt.legend()
    plt.ylim([0,1.5])
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    
    return fig


def plot_safety_label_series(
        m, data_manager, safety_X = None,
        savefile=[False, "test.jpg"],
        display_fig=False
    ):
    fig, ax = plt.subplots(figsize=(8, 4))
    X_explor, U_explor, Y_explor, _ = data_manager.training_pool
    args = np.argsort(X_explor[:,0])
    x = X_explor[args].reshape([-1,1])
    true_z = data_manager.true_safety_function(x)
    
    mu, var = m.predict_f(U_explor[args].reshape([-1, data_manager.return_dimensions('U')]))
    training_u, training_y = m.data
    training_x = training_u[:, training_u.shape[1]//2, None]

    plot_gp(x, mu.numpy(), var.numpy(), "k", "safety GP model")
    plt.plot(training_x, training_y, "kx", label="data")
    
    plt.plot(x, true_z, "k--", label="true Z")
    plt.plot([-1, 1.5], [m.Z_threshold, m.Z_threshold], "g--", label="safety_threshold")
    plt.fill_between( # determined safe region of X
        safety_X, [-10000, -10000], [10000, 10000], color="g", alpha=0.4,
    )
    
    
    plt.title("Safety labels and GP predictions")
    plt.legend()
    plt.ylim([0, 1.5])

    if savefile[0]:
        plt.savefig(savefile[1])

    if not display_fig:
        plt.close(fig)

    return fig


def plot_matrix_with_text(ax, matrix, x_name=None, y_name=None, title='',
                          x_axis=None, y_axis=None, x_ticks_angle=0):
    if x_name is None:
        x_name = np.arange(np.shape(matrix)[1])
    if y_name is None:
        y_name = np.arange(np.shape(matrix)[0])
    
    plt.sca(ax)
    
    for i in range(len(y_name)):
        for j in range(len(x_name)):
            if matrix.dtype == int:
                text = "%d"%(matrix[i, j])
            else:
                text = "%.2f"%(matrix[i, j])
            ax.text(j, i, text,
                    ha="center", va="center", color="w" if matrix[i, j]<=0.8 else "k")
            
    ax.imshow(matrix)
    
    ax.set_xticks(np.arange(len(x_name)))
    ax.set_yticks(np.arange(len(y_name)))
    ax.set_xticklabels(x_name)
    ax.set_yticklabels(y_name)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    
    plt.setp(ax.get_xticklabels(), rotation=x_ticks_angle, ha="right",
             rotation_mode="anchor")
    plt.title(title)


def plot_kernel(kernel_matrix, title='', vmin=None, vmax=None, cmap = 'RdBu_r', savefile=[False,'test.jpg'], display_fig=False):
    shape = np.shape(kernel_matrix)
    vmin = 0 if vmin is None else vmin
    vmax = kernel_matrix.max() if vmax is None else vmax
    nrows = shape[1] if len(shape) == 4 else 1
    ncols = shape[3] if len(shape) == 4 else 1
    
    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle(title)
    
    if len(shape) == 4:
        for i in range(shape[1]):
            for j in range(shape[3]):
                im = axs[i,j].imshow(kernel_matrix[:,i,:,j], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=axs[:,:], location='right', shrink=0.6)
    else:
        im = axs.imshow(kernel_matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=axs, shrink=0.6)
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    
    return fig


def plot_parameter_hist(pars_dict, title='', savefile=[False,'test.jpg'], display_fig=False, bins=20):
    
    if '.kernel.W' in pars_dict:
        P = np.shape(pars_dict['.kernel.W'])[-2]
        L = np.shape(pars_dict['.kernel.W'])[-1]
    else:
        L = len(list(pars_dict.items())) // 2
        P = np.shape(pars_dict['.likelihood.variance'])[-1]
        
    
    nrows = 2 + P
    ncols = L + 1
    
    fig, axs = plt.subplots(
        nrows, ncols,
        #sharex='all', sharey='row',
        figsize=(FIGSIZE_COL*ncols, FIGSIZE_ROW*nrows)
        )
    fig.suptitle(title)
    
    fig.delaxes(axs[0, L])
    fig.delaxes(axs[1, L])
    
    for j in range(L):
        
        key = f'.kernel.kernels[{j}].variance'
        val = pars_dict[key]
        axs[0, j].hist(np.stack(val).flatten(), bins=bins)
        axs[0, j].set_title(key)
        
        key = f'.kernel.kernels[{j}].lengthscales'
        val = pars_dict[key]
        axs[1, j].hist(np.stack(val).flatten(), bins=bins)
        axs[1, j].set_title(key)
        
        key = '.kernel.W'
        for i in range(P):
            if key in pars_dict:
                val = pars_dict[key][..., i,j]
                axs[i+2, j].hist(np.stack(val).flatten(), bins=bins)
                axs[i+2, j].set_title(key+f'[{i},{j}]')
            else:
                fig.delaxes(axs[i+2, j])
            
    for i in range(P):
        key = '.likelihood.variance'
        val = pars_dict[key]
        if len(np.squeeze(val).shape) == 2:
            val = pars_dict[key][..., i]
        axs[2+i, L].hist(np.stack(val).flatten(), bins=bins)
        axs[2+i, L].set_title(key+f'[{i}]')
        
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    
    return fig


def plot_dynamic_trajectories(position, Y, NX_array,
                              style='x-', label=None, color='C0'):
    r"""
    position: [N, ] array
    Y: [N, T] array
    NX_array: [T, ] array, contains only binary numbers
    """
    t = np.tile(position.reshape([1, -1]), [sum(NX_array), 1])
    deltas = -np.arange(len(NX_array), dtype=int)[NX_array==1]
    
    for i, delta in enumerate(deltas):
        t[i, :] += delta
    
    plt.plot(t, Y.T, style, label=label, color=color)
    return True


def get_OWEO_lim(y_pool, scalar = 0.1):
    r"""
    y_pool: [N, D] array
    scalar: float
    
    output: [D, 2] array
    
    we need the exact same ylim across all experiment, which requires some global info
    lim = [..., [min_yi - ai, max_yi + ai],...], ai = scalar * (max_yi - min_yi)
    """
    lower_bound = np.min(y_pool, axis=0)
    upper_bound = np.max(y_pool, axis=0)
    scale = upper_bound - lower_bound
    
    ylim = np.empty([np.shape(y_pool)[1], 2])
    ylim[:, 0] = lower_bound - scalar * scale
    ylim[:, 1] = upper_bound + scalar * scale
    return ylim


def plot_OWEO(m, m_safety, training_pool, test_pool, data_channel_name_tuple, NX_matrix,
              global_title='', color=['C0', 'C3'], savefile=[False, "test.jpg"], display_fig=False):
    
    X_training, U_training, Y_training, Z_training = training_pool
    X_test, U_test, Y_test, Z_test = test_pool
    X_name, U_name, Y_name, Z_name = data_channel_name_tuple
    
    fig, axs = plt.subplots(nrows=max(len(X_name), len(Y_name) + 2), ncols=4, squeeze=False, figsize=(20,max(len(X_name), len(Y_name) + 2)*FIGSIZE_ROW))
    fig.suptitle(global_title)
    
    #iv = m.inducing_variable.inducing_variable.Z.numpy().reshape(-1)
    
    t_tr_pool = np.arange(X_training.shape[0], dtype=int)
    t_test = np.arange(U_test.shape[0], dtype=int)
    if m is None:
        t_tr_used = t_tr_pool[None,:]
    else:
        if len(np.shape(m.history["data_args"])) == 1: # full_task
            t_tr_used = np.sort(m.history["data_args"]).reshape([1,-1])
        else: # partially observed outputs, shape [2, N]
            idx = np.argsort(m.history['data_args'][0,:])
            t_tr_used = m.history['data_args'][:, idx].reshape([2,-1])
    X_tr_used, _, Y_tr_used, Z_tr_used = data_selection(training_pool, t_tr_used)
    
    
    xlim_tr = [t_tr_pool[-1]*(-0.1), t_tr_pool[-1]*1.1]
    ylim_tr = get_OWEO_lim(X_training)
    ylim_te = get_OWEO_lim(X_test)
    ############ subplot [:, 0], subplot [:, 2]
    for i in range(len(X_name)):
        #U_channels = range(sum(sum(NX_matrix[:, :i])), sum(sum(NX_matrix[:, :i+1])))
        
        plt.sca(axs[i, 0])
        # plot_dynamic_trajectories(t_tr_used, U_tr_used[:, U_channels], NX_matrix[:, i],
        #                           style='-', color=color[0])
        plt.plot(t_tr_used[0,:], X_tr_used[:,i], color=color[0])
        plt.title(f"input channel: {X_name[i]}")
        plt.xticks([])
        plt.xlim(xlim_tr) # as t_tr_used keep changing, need to fix the scale
        plt.ylim(ylim_tr[i,:])
        
        plt.sca(axs[i, 2])
        # plot_dynamic_trajectories(t_test, U_test[:, U_channels], NX_matrix[:, i],
        #                           style='-', color=color[0])
        plt.plot(t_test, X_test[:,i], color=color[0])
        plt.title(f"input channel: {X_name[i]}")
        plt.xticks([])
        plt.ylim(ylim_te[i,:])
    
    ############ subplot [:, 1], subplot [:, 3]
    if m is not None:
        entropy = m.entropy(U_training, full_output_cov=True).numpy().reshape(-1)
        mu_tr, var_tr = m.predict_f(U_training)
        mu_tr, var_tr = mu_tr.numpy(), var_tr.numpy()
        
        mu_te, var_te = m.predict_f(U_test)
        mu_te, var_te = mu_te.numpy(), var_te.numpy()
        test_log_lik = m.predict_log_density((U_test, Y_test)).numpy()
    
    ylim_tr = get_OWEO_lim(Y_training)
    ylim_te = get_OWEO_lim(Y_test)
    for i in range(len(Y_name)):
        plt.sca(axs[i, 1])
        plt.plot(t_tr_used[0,:], Y_tr_used[:,i], color=color[0])
        if m is not None:
            plot_gp(t_tr_pool[:, None], mu_tr[:,i,None], var_tr[:,i,None], color[1], None)
        plt.xticks([])
        plt.ylim(ylim_tr[i,:])
        plt.title(f"output channel: {Y_name[i]}")
        
        plt.sca(axs[i, 3])
        plt.plot(t_test, Y_test[:,i], color=color[0])
        if m is not None:
            plot_gp(t_test[:, None], mu_te[:,i,None], var_te[:,i,None], color[1], None)
        plt.xticks([])
        plt.ylim(ylim_te[i,:])
        plt.title(f"output channel: {Y_name[i]}")
    
    ############ subplot [len(Y_name), 1], entropy
    plt.sca(axs[len(Y_name), 1])
    if m is not None:
        plt.plot(t_tr_pool, entropy, color='k')
    plt.xticks([])
    plt.ylim([-15, 5])
    plt.title("entropy")
    
    ############ subplot [len(Y_name), 3], log density
    plt.sca(axs[len(Y_name), 3])
    if m is not None:
        plt.plot(t_test, test_log_lik, color='k')
    plt.xticks([])
    plt.ylim([-170, 10])
    plt.title("log_density")
    
    ############ subplot [len(Y_name)+1, 1], safety constraint
    plt.sca(axs[len(Y_name)+1, 1])
    plt.plot(t_tr_used[0,:], Z_tr_used, color=color[0])
    if m_safety is not None:
        mu_s, var_s = m_safety.predict_f(U_training)
        plot_gp(t_tr_pool[:, None], mu_s.numpy(), var_s.numpy(), color[1], None)
        plt.fill_between(
                axs[len(Y_name)+1, 1].get_xlim(), [-10000, -10000], [m_safety.Z_threshold, m_safety.Z_threshold], color="g", alpha=0.2,label="safe region"
                )
    plt.xlim(axs[len(Y_name)+1, 1].get_xlim())
    plt.ylim(get_OWEO_lim(Z_training)[0,:])
    plt.xticks([])
    plt.title(f"safety constraint: {Z_name[0]}")    
    
    ############ subplot [len(Y_name)+1, 3], NX_matrix
    plt.sca(axs[len(Y_name)+1, 3])
    plot_matrix_with_text(
            axs[len(Y_name)+1, 3], NX_matrix,
            x_name=X_name, y_name=-np.arange(np.shape(NX_matrix)[0], dtype=int), title='NX_matrix',
            y_axis='t', x_ticks_angle=30)
    
    for i in range(len(X_name), len(Y_name)+2):
        fig.delaxes(axs[i, 0])
        fig.delaxes(axs[i, 2])
    
    for i in range(len(Y_name)+2, len(X_name)):
        fig.delaxes(axs[i, 1])
        fig.delaxes(axs[i, 3])
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
    
    return fig


def plot_box(data_matrix, positions, color, plot_outliers: bool=False):
    plt.boxplot(data_matrix, positions=positions, widths = 0.2, notch=True, patch_artist=True,
                boxprops=dict(facecolor=color, color=color),
                capprops=dict(color=color),
                whiskerprops=dict(color=color),
                flierprops=dict(color=color, markeredgecolor=color),
                medianprops=dict(color=color),
                showfliers=plot_outliers
                )


def plot_shaddow(x, y_lower, y_upper, color):
    plt.fill_between(
        x, y_lower, y_upper,
        color=color,
        alpha=0.4,
    )


def plot_RMSE_log_density_global(
        RMSE, log_density,
        xticks=None, xlabel="iterations", ylim=None, yscale = ['linear', 'linear'],
        x_ticks_angle=0, plot_boxes=True, plot_std_error=False,
        significance_test=True, saturation_values=None,
        plot_model = 100,
        savefile=[False, "SafeALvsOthers_RMSE.jpg"],
        display_fig=False
        ):
    RMSE_all = {}
    log_density_all = None if log_density is None else {}
    for name, RMSE_indvidual_model in RMSE.items():
        RMSE_all[name] = np.sqrt(np.mean(
            np.power(RMSE_indvidual_model, 2), axis = -1
            ))[..., None]
        
        if not log_density is None:
            log_dens_indvidual_model = log_density[name]
            log_density_all[name] = log_dens_indvidual_model.sum(axis=-1)[..., None]
    
    return plot_RMSE_log_density(
        RMSE_all, log_density_all,
        xticks = xticks, xlabel = xlabel,
        ylim=ylim, yscale=yscale,
        x_ticks_angle = x_ticks_angle,
        plot_boxes = plot_boxes,
        plot_std_error = plot_std_error,
        significance_test = significance_test,
        saturation_values = saturation_values,
        plot_model = plot_model,
        savefile = savefile, display_fig = display_fig
        )


def plot_RMSE_log_density(RMSE, log_density, func_preds=None, pred_ref=None,
                          xticks=None, xlabel="iterations",
                          ylim=None, yscale = ['linear', 'linear'],
                          x_ticks_angle=0, plot_boxes=True, plot_std_error=False,
                          significance_test=True,
                          saturation_values=None, plot_model = 100,
                          savefile=[False, "SafeALvsOthers_RMSE.jpg"], display_fig=False):

    num_model = min(np.shape(list(RMSE.items()))[0], plot_model)
    _, num_points, num_task = np.shape(list(RMSE.items())[0][1])
    xx = np.arange(num_points)
    xticks_step = max(int(num_points/50), 2)
    if xticks is None:
        xticks = xx
    
    if func_preds is None:
        fig, axs = plt.subplots(nrows=num_task, ncols=2, squeeze=False, figsize=((2*FIGSIZE_COL, num_task*FIGSIZE_ROW)))
    else:
        fig, axs = plt.subplots(nrows=num_task, ncols=num_model+2, squeeze=False, figsize=((num_model + 2)*FIGSIZE_COL, num_task*FIGSIZE_ROW))
        observations_ylim = get_OWEO_lim(pred_ref["True_Y"], 0.2)

    if significance_test:
        xlabel = xlabel+", *: p-value <= 0.05"
    if plot_boxes:
        ylabel = 'mean & boxes'
    elif plot_std_error:
        ylabel = 'mean & std.error'
    else:
        ylabel = 'mean'

    for i in range(num_task):
        #fig.delaxes(axs[row, col])
        for j in range(num_model):
            label, array = list(RMSE.items())[j]
            
            plt.sca(axs[i, 0])
            mean = np.mean(array[...,i], axis=0)
            axs[i, 0].plot(xx, mean, color=f"C{j}", label=label)
            if plot_boxes:
                plot_box(array[...,i], xx + j*0.5/num_model, f"C{j}")
            elif plot_std_error:
                std_err = stats.sem(array[...,i], axis=0)
                #print(f"{label}: mean {mean}, st.error {std_err}")
                plot_shaddow(xx, mean-std_err, mean+std_err, f"C{j}")
            if log_density is not None:
                plt.sca(axs[i, 1])
                axs[i, 1].plot(xx, np.mean(log_density[label][...,i], axis=0), color=f"C{j}", label=label)
                if plot_boxes:
                    plot_box(log_density[label][...,i], xx + j*0.8/num_model, f"C{j}")
                elif plot_std_error:
                    mean = np.mean(log_density[label][...,i], axis=0)
                    std_err = stats.sem(log_density[label][...,i], axis=0)
                    plot_shaddow(xx, mean-std_err, mean+std_err, f"C{j}")
            
            if func_preds is None:
                continue
            plt.sca(axs[i, j+2])
            if pred_ref["X"].shape[-1] == 1:
                plt.plot(pred_ref["X"][..., 0].T, func_preds[label][...,-1,:,i].T, color=f"C{j}")
                plt.plot(pred_ref["X"][..., 0].T, pred_ref["True_Y"][...,i].T, "k--", linewidth=2, label=f"Ground Truth {i+1}")
            else:
                t = np.arange(pred_ref["X"].shape[0], dtype=int)
                plt.plot(t, func_preds[label][...,-1,:,i].T, color=f"C{j}")
                plt.plot(t, pred_ref["True_Y"][...,i].T, "k--", linewidth=2, label=f"Ground Truth {i+1}")
                plt.xticks([])
            plt.title("model predictions, last iteration ## "+label+f"_{i+1}")
            plt.ylim(observations_ylim[i, :])
        if not significance_test:
            plot_bound = [0, 1.0] if ylim is None else ylim[0,i,:]
            axs[i, 0].set_ylim(plot_bound)
        else:
            ### plot stars if significantly different
            plot_bound = [-0.2, 1.0] if ylim is None else [ylim[0,i,0]- 0.2, ylim[0,i,1]]
            axs[i, 0].set_ylim(plot_bound)

            based_label, based_RMSE = list(RMSE.items())[0]
            if not log_density is None:
                base_log_dens = log_density[based_label]
                
            ylim1 = axs[i, 0].get_ylim()
            ylim2 = axs[i, 1].get_ylim()
            
            for j in range(1, num_model):
                label, array = list(RMSE.items())[j]
                iter_RMSE_very_diff = []
                iter_logdens_very_diff = []
                for k in xx:
                    if sum(np.abs(based_RMSE[:,k,i]-array[:,k,i])) > 0:
                        _, p = stats.wilcoxon(based_RMSE[:,k,i], array[:,k,i])
                        if p<=0.05:
                            iter_RMSE_very_diff.append(k)
                    
                    if log_density is None:
                        continue
                    if sum(np.abs(base_log_dens[:,k,i]-log_density[label][:,k,i])) > 0:
                        _, p = stats.wilcoxon(base_log_dens[:,k,i], log_density[label][:,k,i])
                        if p<=0.05:
                            iter_logdens_very_diff.append(k)
                
                axs[i, 0].plot(iter_RMSE_very_diff, np.zeros_like(iter_RMSE_very_diff)-ylim1[1]*0.01*(1+2*j), "*", color=f"C{j}")
                if not log_density is None:
                    axs[i, 1].plot(iter_logdens_very_diff, np.ones_like(iter_logdens_very_diff)*ylim2[0]-(ylim2[1]-ylim2[0])*0.01*(1+2*j), "*", color=f"C{j}")
    ############
        plt.sca(axs[i, 0])
        if not saturation_values is None:
            rmse_ref = saturation_values[0, i]
            plt.plot([xx[0], xx[-1]], [rmse_ref, rmse_ref], 'k--', label=f"value={rmse_ref}")
        plt.legend()#loc='lower right')
        plt.xticks(xx[::xticks_step], xticks[::xticks_step])
        plt.xlabel(xlabel)
        plt.setp(axs[i, 0].get_xticklabels(), rotation=x_ticks_angle, ha="right",
                 rotation_mode="anchor")
        plt.ylabel(f"task{i+1}\n" + ylabel)
        plt.yscale(yscale[0])
        
        plt.sca(axs[i, 1])
        plt.legend()
        plt.xticks(xx[::xticks_step], xticks[::xticks_step])
        plt.xlabel(xlabel)
        #plt.yscale("symlog")
        plt.setp(axs[i, 1].get_xticklabels(), rotation=x_ticks_angle, ha="right",
                 rotation_mode="anchor")
        #plt.ylabel(ylabel)
        plt.yscale(yscale[1])
    
    plt.sca(axs[0, 0])
    plt.title("RMSE")
    plt.sca(axs[0, 1])
    plt.title("log_density curve of groud truth")
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
        
    return fig


def plot_RMSE_log_density_global_diff(
        RMSE, log_density,
        xlabel="iterations", plot_boxes=True,
        savefile=[True, "SafeALvsOthers_RMSE.jpg"],
        display_fig=False
        ):
    RMSE_all = {}
    log_density_all = None if log_density is None else {}
    for name, RMSE_indvidual_model in RMSE.items():
        RMSE_all[name] = np.sqrt(np.mean(
            np.power(RMSE_indvidual_model, 2), axis = -1
            ))[..., None]
        
        if not log_density is None:
            log_dens_indvidual_model = log_density[name]
            log_density_all[name] = log_dens_indvidual_model.sum(axis=-1)[..., None]
    
    return plot_RMSE_log_density_diff(
        RMSE_all, log_density_all,
        xlabel = xlabel,
        plot_boxes = plot_boxes,
        savefile = savefile, display_fig = display_fig
        )


def plot_RMSE_log_density_diff(RMSE, log_density,
                               xlabel="iterations", plot_boxes=True,
                               savefile=[True, "SafeALvsOthers_RMSE.jpg"],
                               display_fig=False):
    
    num_model = np.shape(list(RMSE.items()))[0]
    _, num_points, num_task = np.shape(list(RMSE.items())[0][1])
    xx = np.arange(num_points)
    xticks_step = max(int(num_points/50), 2)
    
    ylabel = 'mean & boxes' if plot_boxes else 'mean'
    
    fig, axs = plt.subplots(nrows=num_task, ncols=2, squeeze=False, figsize=(30,num_task*FIGSIZE_ROW))
    
    for i in range(num_task):
        based_label, based_RMSE = list(RMSE.items())[0]
        if not log_density is None:
            base_log_dens = log_density[based_label]
        
        for j in range(1, num_model):
            label, array = list(RMSE.items())[j]
            color = f"C{j%9}"
            axs[i, 0].plot(xx, np.mean(array[...,i] - based_RMSE[...,i], axis=0), color=color, label=f"- {based_label} + {label}")
            if not log_density is None:
                axs[i, 1].plot(xx, np.mean(base_log_dens[...,i]-log_density[label][...,i], axis=0), color=color, label=f"{based_label} - {label}")
            
            if plot_boxes:
                plt.sca(axs[i, 0])
                plot_box(array[...,i]-based_RMSE[...,i], xx + j*0.5/num_model, color)
                if not log_density is None:
                    plt.sca(axs[i, 1])
                    plot_box(base_log_dens[...,i]-log_density[label][...,i], xx + j*0.5/num_model, color)
            
        plt.sca(axs[i, 0])
        plt.plot([xx[0], xx[-1]], [0, 0], "k--")
        plt.legend()
        plt.xticks(xx[::xticks_step], xx[::xticks_step])
        plt.xlabel(xlabel)
        plt.ylabel(f"task{i+1}\n" + ylabel)
        
        plt.sca(axs[i, 1])
        plt.plot([xx[0], xx[-1]], [0, 0], "k--")
        plt.legend()
        plt.xticks(xx[::xticks_step], xx[::xticks_step])
        #plt.yscale("symlog")
        plt.xlabel(xlabel)
        plt.ylabel(f"task{i+1}\n" + ylabel)
        
    plt.sca(axs[0, 0])
    plt.title("RMSE (the more positive, the better)")
    plt.sca(axs[0, 1])
    plt.title("log_density of groud truth (the more positive, the better)")
    
    if savefile[0]:
        plt.savefig(savefile[1])
    
    if not display_fig:
        plt.close(fig)
        
    return fig


def plot_time(Training_time, DataSelection_time, 
              title=["Training time", "Data selection time"],
              xlabel="iterations", ylabel="seconds",
              savefile=[True, "SafeALvsOthers_times.jpg"],
              ylim={'0':None, '1': None},
              display_fig=False):
    
    num_model = np.shape(list(Training_time.items()))[0]
    _, num_points = np.shape(list(Training_time.items())[0][1])
    xx = np.arange(num_points)
    xticks_step = max(int(num_points/50), 1)
    
    fig = plt.figure(figsize=(15,10))
    
    for i in range(2):
        plt.subplot(2, 1, i+1)
        
        for j in range(num_model):
            if i==0:
                label, array = list(Training_time.items())[j]
                (line,) = plt.plot(xx, np.mean(array, axis=0), label=label)
                plot_box(array, xx + j*0.5/num_model, line.get_color())
            else:
                label, array = list(DataSelection_time.items())[j]
                (line,) = plt.plot(xx[:-1], np.mean(array, axis=0), label=label)
                plot_box(array, xx[:-1] + j*0.5/num_model, line.get_color())
            
            #var = np.var(array, axis=0)
            #plt.fill_between(
            #        xx,
            #        (mean - 2 * np.sqrt(var)),
            #        (mean + 2 * np.sqrt(var)),
            #        alpha=0.4,
            #        )
        plt.title(title[i])
        plt.legend()
        plt.xticks(xx[::xticks_step], xx[::xticks_step])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim[str(i)])
        
    if savefile[0]:
        plt.savefig(savefile[1])
    if not display_fig:
        plt.close(fig)
    return fig



