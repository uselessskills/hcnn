import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from IPython import display
import torch
from hcnn.hcnn import HCNN


class CRCHCNN:
    """
    CRC-HCNN is able to make forecasts only on finite forecast horizon rather than HCCN.
    That's because retro causal net has to be initialized with initial state that is N steps ahead of train data,
    in order to make forecast (backward) on N time stamps after training data.
    """

    def __init__(self, data_dim, hidden_dim=10, init_state_trainable=True, dropout=0, sparsity=0, weight_std=0.1,
                 weight_minmax_norm=None, forecast_horizon=10):
        self.hcnn_c = HCNN(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            init_state_trainable=init_state_trainable,
            dropout=dropout,
            sparsity=sparsity,
            weight_std=weight_std,
            weight_minmax_norm=weight_minmax_norm
        )

        self.hcnn_rc = HCNN(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            init_state_trainable=init_state_trainable,
            dropout=dropout,
            sparsity=sparsity,
            weight_std=weight_std,
            weight_minmax_norm=weight_minmax_norm
        )

        self.forecast_horizon = forecast_horizon

    def train(self, data, state_c=None, state_rc=None, lr=0.001, epochs=10, criterion=torch.nn.MSELoss(),
              reduce_lr_epochs=None, verbose=False, plot_loss=False, plot_pred_train=False):
        opt_c = torch.optim.Adam(self.hcnn_c.hcnn.parameters(), lr=lr)
        opt_rc = torch.optim.Adam(self.hcnn_rc.hcnn.parameters(), lr=lr)
        # criterion = LogCosh.apply  # torch.nn.MSELoss() # + self.hcnn.W.weight.abs().sum()

        state_c = self.hcnn_c.hcnn.init_state if state_c is None else state_c.clone()
        state_rc = self.hcnn_rc.hcnn.init_state if state_rc is None else state_rc.clone()

        lr_c, lr_rc = lr, lr
        losses, losses_c, losses_rc = [], [], []
        for i in range(epochs):
            y_c_sample = self.hcnn_c.sample(state_c, len(data))  # from start to end
            # y_rc_sample = self.hcnn_rc.sample(state_rc, len(data)) # from end to start

            # init_state of rc hcnn is shifted on forecast_horizon so we could make prediction into future on forecast horizon
            y_rc_sample = self.hcnn_rc.sample(state_rc, len(data) + self.forecast_horizon)[
                          self.forecast_horizon:]  # from end to start

            loss_c = self.hcnn_c.bptt(self.hcnn_c.hcnn, state_c, data - y_rc_sample[::-1], opt_c, criterion)
            loss_rc = self.hcnn_rc.bptt(self.hcnn_rc.hcnn, state_rc, data[::-1] - y_c_sample[::-1], opt_rc, criterion)
            losses_c.append(loss_c)
            losses_rc.append(loss_rc)

            # y_crc = self.hcnn_c.sample(state_c, len(data)) + self.hcnn_rc.sample(state_rc, len(data))[::-1]
            y_crc = self.hcnn_c.sample(state_c, len(data)) + self.hcnn_rc.sample(state_rc,
                                                                                 len(data) + self.forecast_horizon)[
                                                             self.forecast_horizon:][::-1]
            loss_crc = criterion(torch.tensor(y_crc).float(), torch.tensor(data).float()) / len(data)
            losses.append(loss_crc)

            if (reduce_lr_epochs is not None) and (i >= reduce_lr_epochs):
                if loss_c > losses_c[i-reduce_lr_epochs]:
                    lr_c /= 2
                for param_group in opt_c.param_groups:
                    param_group['lr'] = lr_c

                if loss_rc > losses_rc[i-reduce_lr_epochs]:
                    lr_rc /= 2
                for param_group in opt_rc.param_groups:
                    param_group['lr'] = lr_rc

            if verbose:
                print(f'Train epoch {i+1}/{epochs}, loss: {loss_crc}')

            if plot_loss and (i + 1) % 25 == 0:
                fig, ax = plt.subplots()
                ax.plot(np.arange(1, i + 2), np.array(losses), 'grey', alpha=0.5, label=f'crc loss: {loss_crc}')
                ax.plot(np.arange(1, i + 2), np.array(losses_c), label=f'causal loss [lr_c = {lr_c}]: {loss_c}')
                ax.plot(np.arange(1, i + 2), np.array(losses_rc), label=f'retro-causal loss [lr_rc = {lr_rc}]: {loss_rc}')
                ax.set_xlabel('epoch'), ax.set_ylabel('loss'), ax.set_title(f'Traning loss on epoch {i+1}')
                legend_loss = ax.legend(frameon=True, loc='upper right')
                legend_loss.get_frame().set_color('white')

                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.close()

            if plot_pred_train and (i + 1) % 25 == 0:
                fig, ax = plt.subplots(figsize=(12, 4), ncols=2)
                ax[0].plot(np.arange(1, i + 2), np.array(losses), 'grey', alpha=0.5, label=f'crc loss: {loss_crc}')
                ax[0].plot(np.arange(1, i + 2), np.array(losses_c), label=f'causal loss [lr_c = {lr_c}]: {loss_c}')
                ax[0].plot(np.arange(1, i + 2), np.array(losses_rc), label=f'retro-causal loss [lr_rc = {lr_rc}]: {loss_rc}')
                ax[0].set_xlabel('epoch'), ax[0].set_ylabel('loss'), ax[0].set_title(f'Traning loss on epoch {i+1}')
                legend_loss = ax[0].legend(frameon=True, loc='upper right')
                legend_loss.get_frame().set_color('white')

                pred = y_crc
                colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
                for c in range(data.shape[1]):
                    ax[1].plot(pred[:, c], '--', label=f'model {c+1}', color=colors[c], alpha=1)
                    ax[1].plot(data[:, c], label=f'actual {c+1}', color=colors[c], alpha=0.3)
                ax[1].set_title(f'predict (start from 0), epoch: {i+1}')
                legend_pred = ax[1].legend(frameon=True, loc='upper right')
                legend_pred.get_frame().set_color('white')

                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.tight_layout()
                plt.close()

        return np.array(losses)

    def sample(self, state_c, state_rc):
        # sample_c = self.hcnn_c.sample(state_c, n)
        # sample_rc = self.hcnn_rc.sample(state_rc, n)
        sample_c = self.hcnn_c.sample(state_c, self.forecast_horizon)
        sample_rc = self.hcnn_rc.sample(state_rc, self.forecast_horizon)
        sample = sample_c + sample_rc[::-1]

        return np.array(sample)