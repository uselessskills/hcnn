import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from IPython import display
from hcnn.hcnn_cell import HCNNCell


class HCNN:
    def __init__(self, data_dim, hidden_dim=10, init_state_trainable=True, dropout=0, sparsity=0, weight_std=0.1,
                 weight_minmax_norm=None):
        self.hcnn = HCNNCell(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            init_state_trainable=init_state_trainable,
            dropout=dropout,
            sparsity=sparsity,
            weight_std=weight_std,
            weight_minmax_norm=weight_minmax_norm
        )

    def init_state(self, bias_std=0.1, bias_minmax_norm=None):
        return self.hcnn.init_state_(bias_std, bias_minmax_norm)

    @staticmethod
    def bptt(hcnn: HCNNCell, state, data, opt, criterion):
        data = torch.tensor(data.copy()).float()

        hcnn.zero_grad()
        s = state
        loss = 0
        for t in range(len(data)):
            y, s = hcnn.forward(s)
            loss += criterion(y, data[t])
            s = s - torch.mm(hcnn.I, (y - data[t]).view(hcnn.data_dim, 1)).view(-1)

        loss.backward()
        opt.step()
        return loss.item() / len(data)

    def train(self, data, state=None, lr=0.001, epochs=10, criterion=torch.nn.MSELoss(), reduce_lr_epochs=None, verbose=False, plot_loss=False,
              plot_pred_train=False):
        # criterion = LogCosh.apply  # torch.nn.MSELoss() # + self.hcnn.W.weight.abs().sum()
        opt = torch.optim.Adam(self.hcnn.parameters(), lr=lr)
        state = self.hcnn.init_state if state is None else state.clone()

        losses = []
        for i in range(epochs):
            loss = self.bptt(self.hcnn, state, data, opt, criterion)
            losses.append(loss)

            if (reduce_lr_epochs is not None) and (i >= reduce_lr_epochs):
                if loss > losses[i-reduce_lr_epochs]:
                    lr /= 2

                for param_group in opt.param_groups:
                    param_group['lr'] = lr


            if verbose:
                print(f'Train epoch {i+1}/{epochs}, lr={lr}, loss: {loss}')

            if plot_loss and (i + 1) % 25 == 0:
                fig, ax = plt.subplots()
                ax.plot(np.arange(1, i + 2), np.array(losses), 'grey')
                ax.set_xlabel('epoch'), ax.set_ylabel('loss'), ax.set_title(f'Traning loss on epoch {i+1}: {loss}')

                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.tight_layout()
                plt.close()

            if plot_pred_train and (i + 1) % 25 == 0:
                fig, ax = plt.subplots(figsize=(12, 4), ncols=2)
                ax[0].plot(np.arange(1, i + 2), np.array(losses), 'grey')
                ax[0].set_xlabel('epoch'), ax[0].set_ylabel('loss'), ax[0].set_title(
                    f'Traning loss on epoch {i+1} [lr={lr}]: {loss}')

                pred = self.sample(state, int(len(data) * 1.0))
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

    def forward(self, state, n):
        s = state.clone()
        for t in range(n):
            y, s = self.hcnn.forward(s)
        return s

    # def backward(self, state, n):  # only for retro causal
    #     s = state.clone()
    #     for t in range(n):
    #         y, s = self.hcnn.backward(s)
    #     return s

    def sample(self, state, n=10):
        sample = []
        s = state.clone()
        for t in range(n):
            y, s = self.hcnn(s)
            sample.append(y.detach().numpy())

        return np.array(sample)


