import contextlib
import logging
import os
from functools import partial, update_wrapper
from importlib import import_module
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
from tqdm import tqdm, trange


class CustomReporter(CLIReporter):
    """Overrides the ray tune's std_out to use logger object"""

    def __init__(self, logger: logging = None, **kwargs):
        super(CustomReporter, self).__init__(**kwargs)
        self.logger = logger

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        self.logger.info(self._progress_str(trials, done, *sys_info))


def convert_to_AR(
    data: torch.Tensor, lag: int = 1, n_steps_ahead: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given the lag, converts time-series (data N x D) into Auto-Regressive format
    ( N-1-lag, D + (D * lag))."""
    if data.ndim == 1:
        data = data.unsqueeze(1)

    window = max(data.shape[0] - (lag + 1), 1)

    if window > 1:
        data_tmp = torch.zeros(
            data.shape[0], ((lag + 1) + n_steps_ahead) * data.shape[1], dtype=data.dtype
        )
        # Create shifted matrix
        ind = 0
        for j in range(data.shape[1]):
            for i in range(lag + 1 + n_steps_ahead):
                data_tmp[:, ind] = torch.roll(
                    data[:, j].unsqueeze(1), shifts=-i, dims=0
                ).squeeze()
                ind += 1

        # Get column idxs for X and Y
        x_n_col_idx = torch.tensor(())
        y_n_col_idx = torch.tensor(())
        for i in range(int(data_tmp.shape[1] / (lag + 1 + n_steps_ahead))):
            start_dx = i * (lag + 1 + n_steps_ahead)
            intermediate_idx = (i * (lag + 1 + n_steps_ahead)) + (lag + 1)
            end_idx = (i + 1) * (lag + 1 + n_steps_ahead)

            x_n_col_idx = torch.cat(
                (x_n_col_idx, torch.arange(start_dx, intermediate_idx).unsqueeze(1))
            )
            y_n_col_idx = torch.cat(
                (y_n_col_idx, torch.arange(intermediate_idx, end_idx).unsqueeze(1))
            )

        return (
            data_tmp[: -(lag + n_steps_ahead), x_n_col_idx.long().squeeze()],
            data_tmp[: -(lag + n_steps_ahead), y_n_col_idx.long().squeeze()],
        )
    else:
        x_n = torch.zeros(data.shape[0], (lag + 1) * data.shape[1], dtype=data.dtype)
        ind = 0
        for j in range(data.shape[1]):
            for i in range(lag + 1):
                x_n[:, ind] = torch.roll(
                    data[:, j].unsqueeze(1), shifts=-i, dims=0
                ).squeeze()
                ind += 1
        return x_n[0, :].unsqueeze(0)


def convert_from_AR(
    X_ar: torch.Tensor = None, Y_ar: torch.Tensor = None, lag: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given the Auto-Regressed data ( N-1-lag, D + (D * lag)), convert into time-series (data N x D)."""

    if Y_ar is not None:
        data = torch.cat((X_ar, Y_ar), dim=1)
    else:
        data = X_ar

    d = int(X_ar.shape[1] / (lag + 1))
    columns = [i + (i * lag) for i in range(d)]

    if d == 1:
        return torch.cat((data[:-1, columns], data[-1, :].unsqueeze(1)), dim=0)
    else:
        op = X_ar[:-1, columns]
        vec = torch.empty(size=(lag + 1, 0))
        for i in range(len(columns)):
            vec = torch.cat(
                (vec, X_ar[-1, columns[i] : columns[i + 1]].unsqueeze(1)), dim=1
            )
            if i == len(columns) - 2:
                vec = torch.cat((vec, X_ar[-1, columns[i + 1] :].unsqueeze(1)), dim=1)
                break

        op = torch.cat((op, vec), dim=0)
        if Y_ar is not None:
            op = torch.cat((op, Y_ar[-1].unsqueeze(0)), dim=0)

        return op


def contains_duplicates(X):
    seen = set()
    seen_add = seen.add
    for x in X:
        if x in seen or seen_add(x):
            return True
    return False


def standardize(x: torch.Tensor, mu=None, std=None):
    if (mu is None) or (std is None):
        mu = x.mean(dim=0)
        std = x.std(dim=0)
    return (x - mu) / std, mu, std


def conditional_tqdm(condition: bool = True, *args, **kwargs):
    if condition:
        return tqdm(*args, **kwargs)
    else:
        return no_op_tqdm(*args, **kwargs)  # do nothing


def conditional_trange(condition: bool = True, *args, **kwargs):
    if condition:
        return trange(*args, **kwargs)
    else:
        return range(*args)  # do nothing


class no_op_tqdm(contextlib.nullcontext):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def update(self, n=1):
        pass


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def str_to_func(func_str):
    modulename, funcname = func_str.rsplit(".", 1)
    mod = import_module(modulename)
    func = getattr(mod, funcname)
    return func


def instantiate_dict_general(config_dict):
    d = {}
    for k, v in config_dict.items():
        d[k] = instantiate(v)
    return d


def instantiate_dict(config_dict):
    d = {}
    for k, v in config_dict.items():
        d[k] = instantiate(v)

    assert (
        len(config_dict.kernel_t.categories) == 1
    ), "kernel_t should only have one category"
    if config_dict.kernel_t.categories[0] == "rbf" and "lag" in config_dict:
        # Don't use lag if KT is 'rbf'
        d.pop("lag")
    elif config_dict.kernel_t.categories[0] != "rbf" and "sigma_t" in config_dict:
        # Don't use sigma_t if KT is not 'rbf'
        d.pop("sigma_t")
    return d


def instantiate(config, *args, is_func=False, _recursive_=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert (
        "_target_" in config
    ), f"Config should have '_target_' for class instantiation but is " + str(config)
    target = config["_target_"]
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit(".", 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # Make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != "_target_"})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs, _recursive_=_recursive_)


def write_yaml(content, fname):
    with fname.open("wt") as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)


def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)


def minimum_in_dict(d, key):
    """
    Find the minimum value in a nested dictionary according to the key.
    """
    return min(d, key=lambda x: d[x][f"{key}"])


def savepdf_tex(filename, warn_on_fail=True):
    """
    Uses inkscape to convert svg to pdf_tex.
    """
    try:
        name = filename.split(".")[0]
        incmd = [
            "inkscape",
            "{}.svg".format(name),
            "--export-filename={}.pdf".format(name),
            "--export-latex",
        ]
        import subprocess
        subprocess.check_output(incmd)
    except Exception as e:
        if warn_on_fail:
            import warnings
            warnings.warn(str(e))
        else:
            raise e


def scatter_plot_with_histogram(
    x: torch.Tensor,
    histogram: bool = False,
    save_path: str = os.getcwd(),
    train_size: int = None,
    title: str = "Latent space",
):
    """
    Scatter plot with histogram.
    :param title:
    :param train_size:
    :param dim:
    :param x:
    :param histogram:
    :param save_path:
    :return: Saves plot in save_path (default= os.getcwd()).
    """
    if x.shape[1] > 2:
        dim = 3
    else:
        dim = 2
    fig = plt.figure()
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        if train_size is not None:
            ax.plot(
                x[:train_size, 0],
                x[:train_size, 1],
                x[:train_size, 2],
                "-",
                linewidth=1,
                c="blue",
                label="Ground-truth",
            )
            ax.plot(
                x[train_size:, 0],
                x[train_size:, 1],
                x[train_size:, 2],
                "-",
                linewidth=1,
                c="green",
                label="Prediction",
            )
        else:
            ax.plot(x[:, 0], x[:, 1], x[:, 2], "-.", c="blue", label="Ground-truth")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # plt.title(f"{title}")
        # plt.legend(loc="upper right")
        plt.legend(loc=(0.6, 0.65))
        plt.tight_layout(pad=0.2)
        plt.savefig(
            f"{save_path}/latent_distribution.svg",
            format="svg",
            dpi=1200,
            transparent=True,
        )
        savepdf_tex(filename=f"{save_path}/latent_distribution.svg")
        plt.show()
    elif dim == 2:
        if not histogram:
            ax = fig.add_subplot(111)
            if train_size is not None:
                ax.plot(
                    x[:train_size, 0],
                    x[:train_size, 1],
                    "-.",
                    c="blue",
                    label="Ground-truth",
                )
                ax.plot(
                    x[train_size:, 0],
                    x[train_size:, 1],
                    "-.",
                    c="green",
                    label="Prediction",
                )
            else:
                ax.plot(x[:, 0], x[:, 1], "-.", c="blue", label="Ground-truth")
        else:
            grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(grid[:-1, 1:])

            if train_size is not None:
                ax.plot(
                    x[:train_size, 0],
                    x[:train_size, 1],
                    "-.",
                    c="blue",
                    label="Ground-truth",
                )
                ax.plot(
                    x[train_size:, 0],
                    x[train_size:, 1],
                    "-.",
                    c="green",
                    label="Ground-truth",
                )
            else:
                ax.plot(x[:, 0], x[:, 1], "-.", c="blue", label="Ground-truth")

            y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=ax)
            x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=ax)

            _, binsx, _ = x_hist.hist(
                x[:, 0], 40, histtype="stepfilled", density=True, orientation="vertical"
            )
            _, binsy, _ = y_hist.hist(
                x[:, 1],
                40,
                histtype="stepfilled",
                density=True,
                orientation="horizontal",
            )
            x_hist.invert_yaxis()
            y_hist.invert_xaxis()
            plt.setp(ax.get_xticklabels(), visible=True)
            plt.setp(ax.get_yticklabels(), visible=True)
        # plt.title(f"{title}")
        plt.legend(loc="upper right")
        plt.tight_layout(pad=0.2)
        plt.savefig(
            f"{save_path}/latent_distribution.svg",
            format="svg",
            dpi=1200,
            transparent=True,
        )
        savepdf_tex(filename=f"{save_path}/latent_distribution.svg")
        plt.show()
    else:
        print("Cannot plot for s = 1")


def animate_trajectory(
    x: torch.Tensor,
    title: str = "Trajectory",
    save_path: str = os.getcwd(),
    train_size: int = None,
):
    x = x.detach().cpu().numpy()

    # THE DATA POINTS
    dataSet = x[:, :3].T  # np.array([x, y, t])
    numDataPoints = dataSet.shape[1]

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = Axes3D(fig)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], "-", c="b", linewidth=1)[
        0
    ]  # For line plot

    dot = plt.plot(dataSet[0], dataSet[1], dataSet[2], "o", c="b")[0]  # For line plot

    # AXES PROPERTIES]
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel("H_x(t)")
    ax.set_ylabel("H_y(t)")
    ax.set_zlabel("H_z(t)")
    ax.set_title(f"{title}")

    # ANIMATION FUNCTION
    def func(num):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, : num + 1])
        line.set_3d_properties(dataSet[2, : num + 1])

        dot.set_data(dataSet[0:2, num])
        dot.set_3d_properties(dataSet[2, num])
        return line, dot

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, func, frames=numDataPoints, interval=15, blit=False
    )
    line_ani.save(f"{save_path}/AnimationNew.mp4")

    plt.show()


def plot_kernel_matrix(
    kernel_matrix, title: str = "Kernel Matrix", save_path: str = os.getcwd()
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.imshow(kernel_matrix, interpolation="nearest")
    plt.tight_layout(pad=0.2)
    plt.savefig(f"{save_path}/{title.replace(' ', '_')}.svg", format="svg", dpi=1200)
    plt.show()


def plot_kernel(kernel_vec, title: str = "Kernel", save_path: str = os.getcwd()):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.plot(kernel_vec)
    plt.grid()
    plt.tight_layout(pad=0.2)
    plt.savefig(f"{save_path}/{title.replace(' ', '_')}.svg", format="svg", dpi=1200)
    plt.show()


def plot_eigenspectrum(
    eigvals,
    lat_space_dim: int,
    title: str = "Eigen Spectrum",
    save_path: str = os.getcwd(),
):
    """
    Plot normalized eigenvalues and show explained variance.
    """
    plt.figure()
    plt.bar(
        range(0, len(eigvals[:lat_space_dim])),
        eigvals[:lat_space_dim],
        alpha=0.5,
        align="center",
        label=r"Eigvals",
    )
    plt.xlabel(f"Principal component index (Upto latent space dim = {lat_space_dim})")
    plt.legend(loc="best")
    plt.tight_layout(pad=0.2)
    plt.savefig(f"{save_path}/{title.replace(' ', '_')}.svg", format="svg", dpi=1200)
    plt.show()