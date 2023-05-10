import logging
from typing import Type

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig

from srcs.model.MV_RKM_nar import Multiview_RKM_NAR
from srcs.utils import instantiate
from srcs.utils.util import savepdf_tex, scatter_plot_with_histogram

plt.style.use("seaborn-bright")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "center")
pd.set_option("display.precision", 3)


class Eval_MV_RKM_nar:
    def __init__(
            self,
            config: DictConfig = None,
            model: Type[Multiview_RKM_NAR] = None,
            train_data: torch.tensor = None,
            test_data: torch.tensor = None,
            Y_pred: torch.tensor = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = model
        self.save_dir = hydra.utils.os.getcwd()
        self.train_data = train_data
        self.test_data = test_data
        self.Y_pred = Y_pred

    def eval_metrics(self):
        """eval_metrics. Computes metrics as defined in config like mse, mae etc"""

        # Init metrics
        metrics = [instantiate(met, is_func=True) for met in self.config["metrics"]]

        scores = []
        for met in metrics:
            scores.append(
                [
                    f"Test error {met.__name__}",
                    met(self.test_data, self.Y_pred.squeeze()).item(),
                ]
            )

        extra_metrics = [
            [
                "Cov H",
                torch.norm(
                    self.model.H_pred.T @ self.model.H_pred
                    - torch.eye(self.model.H_pred.shape[1])
                ).item(),
            ],
            [
                r"Train $||H||_{{f}}^{{2}}$",
                torch.linalg.norm(self.model.H_pred[: self.model.n, :], "fro").item()
                ** 2,
            ],
            [
                r"Test $||H||_{{f}}^{{2}}$",
                torch.linalg.norm(self.model.H_pred[self.model.n:, :], "fro").item()
                ** 2,
            ],
        ]

        df = pd.DataFrame(extra_metrics, columns=["Metrics", "Value"])
        df = df.append(
            pd.DataFrame(scores, columns=["Metrics", "Value"]), ignore_index=True
        )
        self.logger.info("\n {}".format(df))

    def plot_full_data(self):
        """Plot train/test data."""

        _, ax = plt.subplots()
        ax.plot(
            np.arange(0, self.train_data.shape[0]),
            self.train_data,
            "b",
            label="Train",
            linewidth=1,
        )
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.test_data.shape[0],
            ),
            self.test_data,
            "r",
            label="Test",
            linewidth=1,
        )
        ax.set_title("Time series (Full dataset)")
        ax.grid()
        ax.legend(loc="upper right")
        plt.savefig(f"{self.save_dir}/Dataset_full.svg", format="svg", dpi=800)
        plt.show()

    def plot_preds(self):
        # Plot test data
        _, ax = plt.subplots()
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.test_data.shape[0],
            ),
            self.test_data.squeeze(),
            "b",
            label="Ground-truth",
            linewidth=1,
        )
        ax.plot(
            np.arange(
                self.train_data.shape[0],
                self.train_data.shape[0] + self.Y_pred.shape[0],
            ),
            self.Y_pred.squeeze(),
            "g",
            label="Prediction",
            linewidth=1,
        )
        ax.grid()
        plt.tight_layout(pad=0.2)
        ax.legend(loc="upper right")
        plt.savefig(
            f"{self.save_dir}/prediction.svg", format="svg", dpi=1200, transparent=True
        )
        savepdf_tex(filename=f"{self.save_dir}/prediction.svg")
        plt.show()

        if self.test_data.ndim > 1:
            # 3D plot for multi-dimensional data
            ax = plt.figure().add_subplot(projection="3d")
            ax.plot(
                self.test_data[:, 0],
                self.test_data[:, 1],
                self.test_data[:, 2],
                "b",
                label="Ground-truth",
                linewidth=1,
            )
            ax.plot(
                self.Y_pred[:, 0],
                self.Y_pred[:, 1],
                self.Y_pred[:, 2],
                "g",
                label="Prediction",
                linewidth=1,
            )
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.set_zlabel("Z Axis")
            plt.tight_layout(pad=0.2)
            ax.legend()

            plt.savefig(
                f"{self.save_dir}/prediction3D.svg",
                format="svg",
                dpi=1200,
                transparent=True,
            )
            savepdf_tex(filename=f"{self.save_dir}/prediction3D.svg")
            plt.show()

    def plot_latents(self):

        scatter_plot_with_histogram(
            self.model.H_pred,
            histogram=False,
            save_path=self.save_dir,
            train_size=self.model.n,
            title="H of Train, Test set",
        )


    def plot_latent_components(
            self,
            matrix: torch.Tensor = None,
            nr_components: int = None,
    ):
        if matrix is None:
            matrix = self.model.H
        if nr_components is None:
            nr_components = 8  # matrix.shape[1]

        import itertools
        colors = itertools.cycle(['r', 'g', 'b', 'c', 'y', 'm', 'k'])

        fig, ax = plt.subplots(nr_components, 1, sharex=True)
        for i in range(nr_components):
            ax[i].plot(
                matrix[:, i],
                c=next(colors),
                label=f"{i + 1}",
                lw=1,
            )
            ax[i].grid()
            ax[i].legend(loc='upper right')

        plt.xlabel('Time steps')
        plt.tight_layout(pad=0.2)
        plt.savefig(f"{self.save_dir}/latent_components.svg", format="svg", dpi=800)
        savepdf_tex(filename=f"{self.save_dir}/latent_components.svg")
        plt.show()
