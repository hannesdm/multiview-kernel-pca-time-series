import logging

import scipy.linalg as sl
import torch
from torch import nn

from srcs.model.pre_image_method import kernel_smoother, ridge_regression
from srcs.utils.util import conditional_trange, convert_to_AR

logger = logging.getLogger(__name__)


# noinspection PyAttributeOutsideInit
class Multiview_RKM_NAR:
    r"""
    This implements [K(X) + K(Y)] H^T = H^T \Lambda.
    """

    def __init__(self, **kwargs: dict):
        self.kwargs = kwargs
        self.s = self.kwargs["s"]  # Dimension of the latent space.
        self.lag = self.kwargs["lag"]  # time lag
        self.pre_image_method = self.kwargs["pre_image_method"]
        self.pre_image_has_been_fit = False

        # Collect kernel parameters in separate dictionaries.
        self.kwargs_x, self.kwargs_y = {}, {}
        for k, v in self.kwargs.items():
            if k.endswith("_x"):
                self.kwargs_x[k.strip("_x")] = v
            elif k.endswith("_y"):
                self.kwargs_y[k.strip("_y")] = v
            else:
                self.kwargs_x[k] = v
                self.kwargs_y[k] = v

    def train(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Train the model using Recurrent RKM NAR model.
        """
        assert X is not None, "Provide input data matrix"

        if X.ndim == 1:
            X = X.unsqueeze(1)  # Shape the data matrix as n x d
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)  # Shape the data matrix as n x d
        self.d = X.shape[1]
        self.n = X.shape[0]

        self.X = X
        self.Y = Y

        # For X view
        KX = self.compute_kernel(X, **self.kwargs_x)
        if self.kwargs["center_K_x"] is True:
            self.KX_center = center(matrix=KX)
            KX = self.KX_center.centered_matrix

        # For Y view
        KY = self.compute_kernel(Y, **self.kwargs_y)
        if self.kwargs["center_K_y"] is True:
            self.KY_center = center(matrix=KY)
            KY = self.KY_center.centered_matrix

        # Matrix decomposition
        mat = KX + KY

        if self.kwargs["decomposition_method"] == "svd":
            H, S, _ = torch.linalg.svd(mat, full_matrices=True)  # Compute the SVD
            self.lambdas, self.H = S[: self.s], H[:, : self.s]
        elif self.kwargs["decomposition_method"] == "eigen":
            S, H = sl.eigh(mat)  # EigVals are in ascending order
            lambdas = torch.flip(torch.tensor(S[-self.s:]), dims=[0])
            self.H = torch.flip(torch.tensor(H[:, -self.s:]), dims=[1])
        else:
            raise ValueError(
                "Invalid Decomposition method. Valid options are: svd, eigen."
            )

        # Check if the decomposition is successful
        error = torch.norm(
            mat - torch.tensor(H) @ torch.diag(torch.tensor(S)) @ torch.tensor(H).T
        )
        assert error < 1e-5, f"Decomposition error ({error}) more than tolerance!"

        # Pre-compute for later use
        lambdas = torch.diag(lambdas)
        self.coeff_h_y = torch.linalg.inv(lambdas - (self.H.T @ KY @ self.H))
        self.coeff_h_x = torch.linalg.inv(lambdas - (self.H.T @ KX @ self.H))
        if self.kwargs_y["kernel"] == "linear":
            self.coeff_y = self.Y.T @ self.H
        else:
            self.coeff_y = KY @ self.H  # Useful for y_n_ahead
        if self.kwargs_x["kernel"] == "linear":
            self.coeff_x = self.X.T @ self.H
        else:
            self.coeff_x = KX @ self.H  # Useful for x_n_ahead

    def predict(
            self,
            init_vec: torch.tensor = None,
            n_steps: int = 1,
            mode: str = "x_to_y",
            verbose=True,
    ):
        if mode == "x_to_y":
            return self.predict_y(init_vec=init_vec, n_steps=n_steps, verbose=verbose)
        elif mode == "y_to_x":
            return self.predict_x(init_vec=init_vec, n_steps=n_steps, verbose=verbose)
        else:
            raise ValueError("Only 2 modes available: x_to_y and y_to_x")

    def predict_y(self, init_vec: torch.tensor = None, n_steps: int = 1, verbose=True):
        """
        Given X, predicts Y
        """
        if init_vec is None:
            init_vec = convert_to_AR(
                data=self.Y[-(self.lag + 1):],
                lag=self.lag,
                n_steps_ahead=self.kwargs["n_steps_ahead"],
            )
        if init_vec.ndim == 1:
            init_vec = init_vec.unsqueeze(0)

        self.H_pred = self.H.clone()
        self.Y_pred = self.Y.clone()

        for i in conditional_trange(verbose, n_steps, desc="Predicting"):
            if i == 0:
                x_star = init_vec
            else:
                x_star = convert_to_AR(
                    data=self.Y_pred[-(self.lag + 1):],
                    lag=self.lag,
                    n_steps_ahead=self.kwargs["n_steps_ahead"],
                )

            k_x_x = self.compute_kernel(
                X=self.X[: self.n], x=x_star, **self.kwargs_x
            ).type(self.coeff_h_y.type())
            if self.kwargs["center_K_x"] is True:
                k_x_x = self.KX_center(kernel_vector=k_x_x)

            # Predict next hidden variable
            h_n_ahead = self.coeff_h_y @ torch.sum(
                torch.diag(k_x_x) @ self.H_pred[: self.n],
                dim=0,
            )
            self.H_pred = torch.cat((self.H_pred, h_n_ahead.unsqueeze(0)), dim=0)

            # Predict next output variable
            if self.kwargs_y["kernel"] == "linear":
                op = (self.coeff_y @ h_n_ahead).T
            elif self.kwargs["pre_image_method"] == "kernel_smoother":
                # Predict next output variable
                similarities = self.coeff_y @ h_n_ahead
                ks = kernel_smoother(
                    nearest_neighbours=self.kwargs["nearest_neighbours"]
                )
                op = ks(x=self.Y_pred[: self.n], kernel_vector=similarities)
            elif self.kwargs["pre_image_method"] == "ridge_regression":
                if not hasattr(self, "rr"):
                    self.rr = ridge_regression(
                        ridge_regression_alpha=self.kwargs["ridge_regression_alpha"],
                        ridge_regression_rbf_sigma=self.kwargs[
                            "ridge_regression_rbf_sigma"
                        ],
                    )
                op = self.rr(self.Y, self.H_pred)
            else:
                raise ValueError("pre_image_method not valid.")
            if op.ndim < 2:
                op = op.unsqueeze(0)
            self.Y_pred = torch.cat((self.Y_pred, op), dim=0)
        return self.Y_pred[-n_steps:]

    def predict_x(self, init_vec: torch.tensor = None, n_steps: int = 1, verbose=False):
        """
        Given Y, predicts X
        """

        if init_vec.ndim == 1:
            init_vec = init_vec.unsqueeze(0)

        for i in conditional_trange(verbose, n_steps, desc="Predicting"):
            if i == 0:
                y_star = init_vec
            else:
                pass

            k_y_y = self.compute_kernel(
                X=self.Y[: self.n], x=y_star, **self.kwargs_y
            ).type(self.coeff_h_x.type())
            if self.kwargs["center_K_y"] is True:
                k_y_y = self.KY_center(kernel_vector=k_y_y)

            # Predict next hidden variable
            h_n_ahead = self.coeff_h_x @ torch.sum(
                torch.diag(k_y_y) @ self.H[: self.n],
                dim=0,
            )
            self.H = torch.cat((self.H, h_n_ahead.unsqueeze(0)), dim=0)

            # Predict next output variable
            if self.kwargs_x["kernel"] == "linear":
                op = (self.coeff_x @ h_n_ahead).T
            elif self.kwargs["pre_image_method"] == "kernel_smoother":
                # Predict next output variable
                similarities = self.coeff_x @ h_n_ahead
                ks = kernel_smoother(
                    nearest_neighbours=self.kwargs["nearest_neighbours"]
                )
                op = ks(x=self.X[: self.n], kernel_vector=similarities)
            elif self.kwargs["pre_image_method"] == "ridge_regression":
                if not self.rr:
                    self.rr = ridge_regression(
                        ridge_regression_alpha=self.kwargs["ridge_regression_alpha"],
                        ridge_regression_rbf_sigma=self.kwargs[
                            "ridge_regression_rbf_sigma"
                        ],
                    )
                op = self.rr(self.X, self.H)
            else:
                raise ValueError("pre_image_method not valid.")

            self.X = torch.cat((self.X, op.unsqueeze(0)), dim=0)

        return self.X[-n_steps:]

    def compute_kernel(self, X: torch.Tensor = None, x: torch.Tensor = None, **kwargs):
        """
        Compute the kernel matrix or kernel vector.
        Optionally return the pre-defined time-kernel matrix.
        :param X:(optional) (Nxd)
        :param x: (optional) (1xd)
        :return: kernel matrix (N x N) or kernel vector (N x 1)
        """
        # if X is not None:
        if x is None:
            x = X
        elif x.ndim == 1:
            x = x.unsqueeze(1)

        if kwargs["kernel"] == "rbf":
            sqd = torch.cdist(X, x, p=2) ** 2  # memory efficient distance calculations
            return torch.exp(
                -0.5 * sqd / (kwargs["sigma"] ** 2)
            ).squeeze()  # Gaussian Kernel

        elif kwargs["kernel"] == "poly":
            return ((1 + X @ x.T) ** 3).squeeze()

        elif kwargs["kernel"] == "linear":
            return (X @ x.T).squeeze()

        elif kwargs["kernel"] == "laplacian":
            return torch.exp(-X @ x.T / kwargs["sigma"] ** 2).squeeze()

        elif kwargs["kernel"] == "sigmoid":
            return torch.tanh(X @ x.T / kwargs["sigma"]).squeeze()

        elif kwargs["kernel"] == "cosine":
            return torch.nn.functional.cosine_similarity(X, x).squeeze()

        elif kwargs["kernel"] == "lin_decay":
            #  Create the pre-defined Time kernel matrix. It does not require T.
            assert self.kwargs["lag"] is not None, (
                "Provide a positive integer lag. Required for pre-defined time kernel matrix.\n"
                "Another option is to use an available kernel-function to define a time kernel matrix."
            )
            assert (
                    self.n > self.kwargs["lag"] > 0
            ), "Number of samples must be greater than the lag."
            self.lag = self.kwargs["lag"]

            coeff = self.lag * torch.eye(self.n)
            for _p in range(1, self.lag + 1):
                o = (self.lag - _p) * torch.ones((self.n - _p))
                coeff += torch.diag(o, diagonal=_p)
                coeff += torch.diag(o, diagonal=-_p)
            return (1 / self.lag) * coeff

        elif kwargs["kernel"] == "step_decay":
            #  Create the pre-defined Time kernel matrix. It does not require T.
            assert self.kwargs["lag"] is not None, (
                "Provide a positive integer lag. Required for pre-defined time kernel matrix.\n"
                "Another option is to use an available kernel-function to define a time kernel matrix."
            )
            assert (
                    self.n > self.kwargs["lag"] > 0
            ), "Number of samples must be greater than the lag."
            self.lag = self.kwargs["lag"]

            coeff = torch.eye(self.n)
            for _p in range(1, self.lag + 1):
                o = (1 / (2 * self.lag)) * torch.ones((self.n - _p))
                coeff += torch.diag(o, diagonal=_p)
                coeff += torch.diag(o, diagonal=-_p)
            return coeff

        elif kwargs["kernel"] == "indicator":
            # Following is not a kernel. It is an indicator function. Use eigen-decomposition instead.
            if self.kwargs["lag"] > self.n:
                logger.info("Lag must be <  # of samples. Setting lag = n-1.")
            self.lag = min(self.n - 1, self.kwargs["lag"])

            if self.kwargs["decomposition_method"] == "svd":
                logger.info(
                    "Indicator function is not a kernel. Using eigen-decomposition instead."
                )
                self.kwargs["decomposition_method"] = "eigen"

            coeff = torch.eye(self.n)
            for _p in range(1, self.lag + 1):
                o = torch.ones((self.n - _p))
                coeff += torch.diag(o, diagonal=_p)
                coeff += torch.diag(o, diagonal=-_p)
            return coeff

        elif kwargs["kernel"] == "diag_0_indicator":
            # Following is not a kernel. It is a indicator function. Use eigen-decomposition instead.
            if self.kwargs["lag"] > self.n:
                logger.info("Lag must be <  # of samples. Setting lag = n-1.")
            self.lag = min(self.n - 1, self.kwargs["lag"])

            if self.kwargs["decomposition_method"] == "svd":
                logger.info(
                    "Indicator function is not a kernel. Using eigen-decomposition instead."
                )
                self.kwargs["decomposition_method"] = "eigen"

            coeff = torch.zeros(self.n, self.n)
            for _p in range(1, self.lag + 1):
                o = torch.ones((self.n - _p))
                coeff += torch.diag(o, diagonal=_p)
                coeff += torch.diag(o, diagonal=-_p)
            return coeff

        else:
            raise ValueError(
                "Invalid kernel. Valid kernels are: rbf, poly, linear, laplacian, sigmoid, cosine, lin_decay, step."
            )

    def evaluate_objective(self, H: torch.Tensor = None):
        pass

    def evaluate_gradient(self, H: torch.Tensor = None):
        pass


class center:
    """
    Center the matrix or feature vector.
    """

    def __init__(self, matrix: torch.Tensor = None):
        n = matrix.shape[0]
        one_n_mat = torch.ones((n, n), dtype=matrix.dtype)
        eye = torch.eye(n, dtype=matrix.dtype)

        self.coeff_center_mat = eye - (one_n_mat / n)
        self.centered_matrix = self.coeff_center_mat @ matrix @ self.coeff_center_mat

        # Pre-compute the terms needed for centering kernel vector.
        # Used in the prediction loops.
        self.coeff_vec = (
                ((one_n_mat / (n ** 2)) - (eye / n))
                @ self.centered_matrix
                @ torch.ones(n, dtype=matrix.dtype)
        )

    def __call__(self, kernel_vector: torch.Tensor = None):
        return (self.coeff_center_mat @ kernel_vector) + self.coeff_vec
