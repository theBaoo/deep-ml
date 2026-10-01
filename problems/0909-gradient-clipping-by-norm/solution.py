import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    # TODO: compute total grad norm, scale in-place if it exceeds max_norm, return original norm
    parameters = list(parameters)

    grads = [
        p.grad.flatten()
        for p in parameters
        if p.grad is not None
    ]

    if not grads:
        return 0.0 

    norm = torch.linalg.vector_norm(
        torch.cat(grads)
    )

    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)

        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)


    return norm.item()
