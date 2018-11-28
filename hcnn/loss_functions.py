import torch


class LogCosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y):
        ctx.save_for_backward(y_pred, y)
        return torch.log(torch.cosh(y_pred - y)).sum()

    @staticmethod
    def backward(ctx, grad_output):
        yy_pred, yy = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.tanh(yy_pred - yy)
        return grad_input, None