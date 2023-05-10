"""
This file implements various metrics to evaluate the performance of the model.
"""
import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def MSE(output, actual):
    """
    Squared-L2 norm
    = (1/N) \sum_{i=1}^{N} || Y^{\hat}_{i} - Y_{i} ||_{2}^{2}
    """
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    loss = torch.nn.MSELoss(reduction='sum')  # see pytorch documentation on MSELoss
    return loss(output.squeeze(), actual.squeeze()) / output.squeeze().shape[0]


def MAE(output, actual):
    r"""
    Squared-L1 norm
      = (1/N) \sum_{i=1}^{N} || Y^{\hat}_{i} - Y_{i} ||_{1}^{2}
    """
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    loss = torch.nn.L1Loss(reduction='sum')  # see pytorch documentation on MAELoss
    return loss(output.squeeze(), actual.squeeze()) / output.squeeze().shape[0]


def PearsonR(x, y, batch_first=False, min=True):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)
    if min:
        corr *= -1
    return corr


def huberloss(output, actual):
    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    loss = torch.nn.HuberLoss(reduction='sum')
    return loss(output.squeeze(), actual.squeeze()) / output.squeeze().shape[0]


def diag_error(M: torch.Tensor = None, plot: bool = False, title: str = 'M'):
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(M)
        ax.set_title(f'{title}')
        plt.show()
    return torch.norm(M - torch.diag(torch.diag(M)), 'fro') / torch.norm(M, 'fro')
