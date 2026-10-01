import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        mask = x > 0
        ctx.save_for_backward(mask)
        return torch.where(x > 0, x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        mask, = ctx.saved_tensors
        return grad_output * mask
