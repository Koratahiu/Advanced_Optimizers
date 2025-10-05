import torch
from torch.optim import Optimizer
from typing import Callable

class KourkoutasHelper:
    """
    A helper class to add layer-wise Kourkoutas-β functionality to a PyTorch optimizer.
    """
    def __init__(self, optimizer: Optimizer):
        # We need a reference to the optimizer to access its param_groups and state
        if not hasattr(optimizer, 'param_groups'):
            raise TypeError("optimizer must be a valid torch.optim.Optimizer instance.")
        self.optimizer = optimizer
        
        # State managed by the helper
        self.layer_state = {}
        self.layer_info = {}
        self._layer_info_built = False
        self._current_step_prepared = -1

    def _build_layer_info_if_needed(self):
        """Builds a map of layers and the parameters they contain."""
        if self._layer_info_built:
            return
            
        if not hasattr(self.optimizer, 'layer_key_fn') or self.optimizer.layer_key_fn is None:
            print("Warning: KourkoutasHelper requires 'layer_key_fn' on the optimizer. Defaulting to tensor-wise (id).")
            self.optimizer.layer_key_fn = lambda p: id(p)

        for group in self.optimizer.param_groups:
            if not group.get('kourkoutas_beta', False):
                continue
            for p in group['params']:
                if p.grad is None: continue
                layer_key = self.optimizer.layer_key_fn(p)
                if layer_key not in self.layer_info:
                    self.layer_info[layer_key] = {'params': [], 'group_ref': group}
                self.layer_info[layer_key]['params'].append(p)
        self._layer_info_built = True

    def prepare_step(self):
        """
        Calculates dynamic beta2 for all layers using the completed scalar accumulators
        from the PREVIOUS step. Should be called once at the start of an optimizer step.
        """
        self._build_layer_info_if_needed()

        if hasattr(self.optimizer, 'logging') and self.optimizer.logging:
            if not hasattr(self.optimizer, '_beta2_log'):
                self.optimizer._beta2_log = []

        for layer_key, info in self.layer_info.items():
            params, group = info['params'], info['group_ref']
            
            if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                    'r_ema_grad_norm': torch.tensor(0.0, device=params[0].device, dtype=torch.float32),
                    'sum_sq_accumulator': torch.tensor(0.0, device=params[0].device, dtype=torch.float32)
                }
            
            layer_state = self.layer_state[layer_key]
            
            pooled_grad_norm = torch.sqrt(layer_state['sum_sq_accumulator'])
            
            r_ema = layer_state['r_ema_grad_norm']
            r_ema.mul_(group['ema_alpha']).add_(pooled_grad_norm, alpha=1.0 - group['ema_alpha'])
            
            raw = pooled_grad_norm / (r_ema + group['tiny_spike'])
            sun = raw / (1.0 + raw)
            beta2_max = group['betas'][1]
            beta2 = beta2_max - (beta2_max - group['beta2_min']) * sun
            
            layer_state['dynamic_beta2'] = beta2.item()
            layer_state['sum_sq_accumulator'].zero_()

            if hasattr(self.optimizer, 'logging') and self.optimizer.logging and hasattr(self.optimizer, '_beta2_log'):
                self.optimizer._beta2_log.append(beta2.item())

    def maybe_prepare_step(self, current_step: int):
        """
        A universal guard that calls prepare_step() exactly once per training step.
        """
        if self._current_step_prepared < current_step:
            self.prepare_step()
            self._current_step_prepared = current_step

    def accumulate_gradient_sq_norm(self, p: torch.Tensor, grad: torch.Tensor):
        """
        Accumulates the squared L2 norm of a single gradient for the next step's calculation.
        """
        layer_key = self.optimizer.layer_key_fn(p)
        if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                'r_ema_grad_norm': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'sum_sq_accumulator': torch.tensor(0.0, device=p.device, dtype=torch.float32)
            }
        self.layer_state[layer_key]['sum_sq_accumulator'] += torch.sum(grad.detach().pow(2)).float()

    def get_beta2(self, p: torch.Tensor, group: dict, current_step: int) -> float:
        """
        Gets the appropriate beta2 for the current parameter, handling warmup and dynamic value fetching.
        """
        beta2_default = group['betas'][1]
        if current_step < group['k_warmup_steps']:
            return 0.5 * (group['beta2_min'] + beta2_default)
        
        layer_key = self.optimizer.layer_key_fn(p)
        return self.layer_state.get(layer_key, {}).get('dynamic_beta2', beta2_default)