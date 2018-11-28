import torch


def min_max_norm(x, range_=(-1, 1)):
  scaled_01 = (x - torch.min(x))/(torch.max(x) - torch.min(x))
  return scaled_01*(range_[1] - range_[0]) + range_[0]


class HCNNCell(torch.nn.Module):
    def __init__(self, data_dim, hidden_dim=10, init_state_trainable=True, dropout=0, sparsity=0, weight_std=0.1,
                 weight_minmax_norm=None):
        super(HCNNCell, self).__init__()

        self.init_state = None
        if init_state_trainable:
            self.init_state = torch.nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            torch.nn.init.normal_(self.init_state, std=weight_std)

        self.W = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        torch.nn.init.sparse_(self.W.weight, sparsity=sparsity,
                              std=weight_std)  # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.init.sparse

        if weight_minmax_norm is not None:
            self.W.weight.data = min_max_norm(self.W.weight.data, (-weight_minmax_norm, weight_minmax_norm))
            if self.init_state is not None:
                self.init_state.weight.data = min_max_norm(self.init_state.weight.data,
                                                           (-weight_minmax_norm, weight_minmax_norm))

        self.dropout = torch.nn.Dropout(dropout)
        self.I = torch.cat((torch.eye(data_dim), torch.zeros(hidden_dim - data_dim, data_dim)), dim=0)
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

    def forward(self, s):
        s = torch.tanh(self.dropout(self.W(s)))
        y = torch.mm(self.I.transpose(1, 0), s.view(self.hidden_dim, 1)).view(-1)  # s[t][:data_dim]
        return y, s

    # def backward(self, s):  # only for retro causal
    #     # if s2 = tanh(W*s1), then s1 = inv(W)*tanh_inv(s2)
    #     s = torch.mm(self.W.weight.data.inverse(),
    #                  torch.log((1 + s.data) / (1 - s.data)).view(s.size()[0], 1) / 2).view(-1)
    #     y = torch.mm(self.I.transpose(1, 0), s.view(self.hidden_dim, 1)).view(-1)  # s[t][:data_dim]
    #     return y, s

    def init_state_(self, bias_std=0.1, bias_minmax_norm=None):
        bias = torch.randn(self.hidden_dim) * bias_std
        if bias_minmax_norm is not None:
            bias = min_max_norm(bias, (-bias_minmax_norm, bias_minmax_norm))
        return bias