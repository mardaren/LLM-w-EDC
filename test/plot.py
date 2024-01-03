import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt


matplotlib.rcParams.update({'font.size': 16})


def plot_colormap(model, x_data, y_data, hidden_size):
    col = np.linspace(-5.5, 5, 200)
    a, b = np.meshgrid(col, col)
    x = np.array([a.flatten(), b.flatten()]).T
    _, y_proba, u = model.get_predictions(torch.Tensor(x))
    probs = y_proba[:, 1].reshape(200, 200)
    uncertainty = u.reshape(200, 200)

    x_test_0 = x_data[np.where(y_data[:, 0] == 1)[0]]
    x_test_1 = x_data[np.where(y_data[:, 1] == 1)[0]]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cp = axs[0].contourf(a, b, probs, levels=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1])
    fig.colorbar(cp)
    axs[0].plot(x_test_0[:, 0], x_test_0[:, 1], 'r.')
    axs[0].plot(x_test_1[:, 0], x_test_1[:, 1], 'k.')
    axs[0].set_title(f'Probabilities')
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')

    cp = axs[1].contourf(a, b, uncertainty, levels=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1], cmap="RdPu")
    fig.colorbar(cp)
    axs[1].set_title(f'Uncertainties')
    axs[1].set_xlabel('x1')
    # axs[1].set_ylabel('x2')

    plt.savefig(f'test/results/probabilities_uncertainty_basic.png')
    plt.show()


def plot_losses(losses, hidden_size):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title(f'Training Results: {hidden_size} Hidden Units')
    plt.savefig(f'training_result_{hidden_size}.png')
    plt.show()
