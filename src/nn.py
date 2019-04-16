import torch
import math


def sample_gumbel(shape):
    uniform_samples = torch.empty(*shape).uniform_(1e-5, 1. - 1e-5)
    return -torch.log(-torch.log(uniform_samples))


def sample_masked_multinomial(logits, mask, axis=None):
    gumbels = sample_gumbel(logits.shape)
    noisy_logits = logits + gumbels
    min_val: torch.Tensor = torch.min(noisy_logits) - 1.
    min_val = min_val.expand_as(logits)
    masked_logits = torch.where(mask, min_val, noisy_logits)
    return torch.argmax(masked_logits, dim=axis)


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    one_hot = torch.Tensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.view(-1, 1), 1)

    return target
