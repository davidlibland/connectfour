from functools import cache

import torch


@cache
def get_winning_filters(run_length: int = 4):
    # get horizontal filter
    horiz = torch.einsum(
        "ij,kl->ijkl", torch.eye(3), torch.ones([1, run_length])
    )
    # get vertical filter
    vert = torch.einsum(
        "ij,kl->ijkl", torch.eye(3), torch.ones([run_length, 1])
    )
    # get diagonal filter
    diag = torch.einsum("ij,kl->ijkl", torch.eye(3), torch.eye(run_length))
    # get anti-diagonal filter
    anti_diag_ = torch.flip(torch.eye(run_length), (1,))
    anti_diag = torch.einsum("ij,kl->ijkl", torch.eye(3), anti_diag_)
    return [horiz, vert, diag, anti_diag]
