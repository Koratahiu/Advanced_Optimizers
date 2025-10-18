import torch
from torch.optim import Optimizer
from typing import Callable, Optional

class MuonAdamHelper:
    """
    A helper class for Muon_adv to decide whether to use Muon or a delegate
    AdamW optimizer for a given parameter based on a keying function.
    """
    def __init__(self, optimizer: Optimizer, layer_key_fn: Optional[Callable]):
        if not hasattr(optimizer, 'param_groups'):
            raise TypeError("optimizer must be a valid torch.optim.Optimizer instance.")
        self.optimizer = optimizer
        
        if layer_key_fn is None:
            # If no function is provided, default all parameters to 'muon'.
            self.layer_key_fn = lambda p: 'muon'
        else:
            self.layer_key_fn = layer_key_fn

    def get_optimizer_type(self, p: "torch.Tensor") -> str:
        """
        Returns the designated optimizer type ('adam' or 'muon') for a parameter.
        
        The user-provided layer_key_fn should return 'adam' for parameters
        to be handled by the auxiliary AdamW optimizer. Any other return
        value is treated as 'muon'.
        """
        key = self.layer_key_fn(p)
        if key == 'adam':
            return 'adam'
        return 'muon'