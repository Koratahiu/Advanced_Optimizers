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

        # This ensures the map is complete before the first backward pass,
        # making it compatible with fused back pass mechanisms.
        self._build_layer_info_if_needed()

    def _build_layer_info_if_needed(self):
        """Builds a map of layers and the parameters they contain."""
        if self._layer_info_built:
            return
            
        if not hasattr(self.optimizer, 'layer_key_fn') or self.optimizer.layer_key_fn is None:
            print("Warning: KourkoutasHelper requires 'layer_key_fn' on the optimizer. Defaulting to tensor-wise (id).")
            self.optimizer.layer_key_fn = lambda p: id(p)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                # The mapping is static and should not depend on the presence of a gradient.
                layer_key = self.optimizer.layer_key_fn(p)
                if layer_key not in self.layer_info:
                    self.layer_info[layer_key] = {'params': [], 'group_ref': group}
                self.layer_info[layer_key]['params'].append(p)
        
        k_logging_interval = self.optimizer.param_groups[0].get('k_logging', 0)
        if k_logging_interval > 0:
            print(f"[Kourkoutas-β Debug] Layer info built. Found {len(self.layer_info)} unique layers/buckets.")

        self._layer_info_built = True

    def prepare_step(self, current_step: int):
        """
        Calculates dynamic beta2 for all layers using the completed scalar accumulators
        from the PREVIOUS step. Should be called once at the start of an optimizer step.
        """

        # Check if logging is enabled for this step based on the interval
        k_logging_interval = self.optimizer.param_groups[0].get('k_logging', 0)
        is_logging_step = k_logging_interval > 0 and (current_step + 1) % k_logging_interval == 0

        beta2_log = [] if is_logging_step else None
        first_layer_key = next(iter(self.layer_info), None)

        for layer_key, info in self.layer_info.items():
            params, group = info['params'], info['group_ref']
            
            if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                    'r_ema_grad_norm': torch.tensor(0.0, device=params[0].device, dtype=torch.float32),
                    'sum_sq_accumulator': torch.tensor(0.0, device=params[0].device, dtype=torch.float32)
                }
            
            layer_state = self.layer_state[layer_key]
            
            # Use the completed accumulator from the previous step
            pooled_grad_norm = torch.sqrt(layer_state['sum_sq_accumulator'])
            
            r_ema = layer_state['r_ema_grad_norm']
            prev_r_ema_val = r_ema.item() # for logging
            
            # EMA is always updated, even during warmup
            r_ema.mul_(group['ema_alpha']).add_(pooled_grad_norm, alpha=1.0 - group['ema_alpha'])
            
            sun = torch.tensor(0.0, device=r_ema.device) # Default sun to 0 for warmup
            beta2_max = group['betas'][1]

            # --- CONSOLIDATED WARMUP LOGIC ---
            if current_step < group['k_warmup_steps']:
                beta2 = beta2_max
            else:
                raw = pooled_grad_norm / (r_ema + group['tiny_spike'])
                sun = raw / (1.0 + raw)
                beta2 = beta2_max - (beta2_max - group['beta2_min']) * sun

            layer_state['dynamic_beta2'] = beta2.item() if isinstance(beta2, torch.Tensor) else beta2
            layer_state['sum_sq_accumulator'].zero_()

            if is_logging_step:
                beta2_log.append(layer_state['dynamic_beta2'])
                if layer_key == first_layer_key:
                    print(f"\n[Kourkoutas-β Debug] Step {current_step + 1} - Sample Layer '{layer_key}':")
                    print(f"  - Grad Norm: {pooled_grad_norm.item():.4e}, Prev EMA: {prev_r_ema_val:.4e}, New EMA: {r_ema.item():.4e}")
                    print(f"  - Sunspike: {sun.item():.4f}, Dynamic Beta2: {layer_state['dynamic_beta2']:.4f}")
        
        if is_logging_step and beta2_log:
            beta2_tensor = torch.tensor(beta2_log, device='cpu')
            print(f"[Kourkoutas-β Debug] Step {current_step + 1} Overall Beta2 Stats: Min={beta2_tensor.min():.4f}, Max={beta2_tensor.max():.4f}, Mean={beta2_tensor.mean():.4f}")


    def maybe_prepare_step(self, current_step: int):
        """
        A universal guard that calls prepare_step() exactly once per training step.
        """
        if self._current_step_prepared < current_step:
            self.prepare_step(current_step)
            self._current_step_prepared = current_step

    def accumulate_gradient_sq_norm(self, p: torch.Tensor, grad: torch.Tensor):
        """
        Accumulates the squared L2 norm of a single gradient for the next step's calculation.
        """
        layer_key = self.optimizer.layer_key_fn(p)

        if layer_key in self.layer_info:
            if layer_key not in self.layer_state:
                    self.layer_state[layer_key] = {
                    'r_ema_grad_norm': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                    'sum_sq_accumulator': torch.tensor(0.0, device=p.device, dtype=torch.float32)
                }
            # Accumulate for the *next* step's prepare_step call
            self.layer_state[layer_key]['sum_sq_accumulator'] += torch.sum(grad.detach().pow(2)).float()

    def get_beta2(self, p: torch.Tensor, group: dict, current_step: int) -> float:
        """
        Gets the appropriate beta2 for the current parameter, handling warmup and dynamic value fetching.
        """
        layer_key = self.optimizer.layer_key_fn(p)
        # The default is the max value, which is correct for unmapped params or edge cases
        return self.layer_state.get(layer_key, {}).get('dynamic_beta2', group['betas'][1])