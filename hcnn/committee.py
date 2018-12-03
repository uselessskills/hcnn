from multiprocessing import Pool
from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from IPython import display
from hcnn.crchcnn import CRCHCNN


class CRCHCNNCommittee:
    def __init__(self, data_dim, hidden_dim=10, dropout=0, sparsity=0,
                 weight_std=0.1, weight_minmax_norm=None, forecast_horizon=10, n_estimators=10):

        self.committee = [None] * n_estimators
        self.forecast_horizon = forecast_horizon
        for i in range(n_estimators):
            self.committee[i] = CRCHCNN(
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                init_state_trainable=True,
                dropout=dropout,
                sparsity=sparsity,
                weight_std=weight_std,
                weight_minmax_norm=weight_minmax_norm,
                forecast_horizon=forecast_horizon
            )

    @staticmethod
    def train_(data, lr, epochs, criterion, reduce_lr_epochs, crchcnn):
        return crchcnn.train(data=data, epochs=epochs, lr=lr, criterion=criterion,
                             reduce_lr_epochs=reduce_lr_epochs)

    def train(self, data, lr=0.001, epochs=10, criterion=torch.nn.MSELoss(),
              reduce_lr_epochs=None, verbose=False, plot_loss=False, plot_pred_train=False, n_processes=None):

        losses = []
        for i in range(epochs):
            train_func = partial(self.train_, data, lr, 1, criterion, reduce_lr_epochs)
            with Pool(len(self.committee) if n_processes is None else n_processes) as p:
                loss = p.map(train_func, self.committee)
                losses.append(np.array(loss).flatten())

                if verbose:
                    print(f'Train epoch {i+1}/{epochs}, loss: {np.array(loss).flatten().mean()}')

                if plot_loss and (i + 1) % 5 == 0:
                    fig, ax = plt.subplots()
                    ax.plot(np.arange(1, i + 2), np.array(losses).mean(axis=1), 'grey', alpha=1.0,
                            label=f'committee loss: {np.array(loss).flatten().mean()}')
                    for k in range(len(self.committee)):
                        ax.plot(np.arange(1, i + 2), np.array(losses)[:, k], 'grey', alpha=0.2)
                    ax.set_xlabel('epoch'), ax.set_ylabel('loss'), ax.set_title(f'Traning loss on epoch {i+1}')
                    legend_loss = ax.legend(frameon=True, loc='upper right')
                    legend_loss.get_frame().set_color('white')

                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                    plt.close()

                if plot_pred_train and (i + 1) % 5 == 0:
                    fig, ax = plt.subplots(figsize=(12, 4), ncols=2)
                    ax[0].plot(np.arange(1, i + 2), np.array(losses).mean(axis=1), 'grey', alpha=1.0,
                            label=f'committee loss: {np.array(loss).flatten().mean()}')
                    for k in range(len(self.committee)):
                        ax[0].plot(np.arange(1, i + 2), np.array(losses)[:, k], 'grey', alpha=0.2)
                    ax[0].set_xlabel('epoch'), ax[0].set_ylabel('loss'), ax[0].set_title(f'Traning loss on epoch {i+1}')
                    legend_loss = ax[0].legend(frameon=True, loc='upper right')
                    legend_loss.get_frame().set_color('white')

                    pred_mean, pred_committee = self.sample()
                    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
                    for c in range(data.shape[1]):
                        ax[1].plot(pred_mean[:, c], '--', label=f'model {c+1}', color=colors[c], alpha=1)
                        for k in range(len(self.committee)):
                            ax[1].plot(pred_committee[k, :, c], '--', color=colors[c], alpha=0.2)
                        ax[1].plot(data[:, c], label=f'actual {c+1}', color=colors[c], alpha=0.5)
                    ax[1].set_title(f'predict (start from 0), epoch: {i+1}')
                    legend_pred = ax[1].legend(frameon=True, loc='upper right')
                    legend_pred.get_frame().set_color('white')

                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                    plt.tight_layout()
                    plt.close()

        return np.array(losses)

    @staticmethod
    def sample_(on_train, crchcnn):
        if on_train:
            state_c = crchcnn.hcnn_c.hcnn.init_state
            state_rc = crchcnn.hcnn_rc.forward(crchcnn.hcnn_rc.hcnn.init_state, crchcnn.forecast_horizon)
        else:
            state_c = crchcnn.hcnn_c.forward(crchcnn.hcnn_c.hcnn.init_state, crchcnn.forecast_horizon)
            state_rc = crchcnn.hcnn_rc.hcnn.init_state

        return crchcnn.sample(state_c=state_c, state_rc=state_rc)

    def sample(self, on_train=False):
        with Pool(len(self.committee)) as p:
            sample_func = partial(self.sample_, on_train)
            samples = np.array(p.map(sample_func, self.committee))
        return samples.mean(axis=0), samples



