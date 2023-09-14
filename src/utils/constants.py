from itertools import product

import torch

BOX_OFFSETS: torch.Tensor


def declare_globals() -> None:
    """This function declares a global tensors which will be accessed frequently. This will reduce repetitive cuda memory allocation.
    We can initialize tensors on the `default` device by invoking this function after the `torch.set_default_device` statement.
    """
    global BOX_OFFSETS
    BOX_OFFSETS = torch.tensor(list(product([0, 1], [0, 1], [0, 1])))
