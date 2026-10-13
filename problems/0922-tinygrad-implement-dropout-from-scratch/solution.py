from tinygrad import Tensor

def dropout(x, p, training):
    # TODO: inverted dropout
    if not training:
        return x

    if p == 0:
        return x

    logits = Tensor.rand(*x.shape)
    mask = logits >= p

    return x * mask / (1 - p)

    
