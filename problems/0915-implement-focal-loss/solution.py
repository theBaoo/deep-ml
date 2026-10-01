import torch

def focal_loss(logits, targets, gamma=2.0):
    # TODO: mean focal loss over the batch
    probs = torch.softmax(logits, dim=-1)
    p_t = probs[torch.arange(logits.size(0)), targets]
    return (-(1 - p_t) ** gamma * torch.log(p_t)).mean()
