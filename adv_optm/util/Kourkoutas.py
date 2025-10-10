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
        self.layer_state = {}

        self.layer_info = {}
        self._layer_info_built = False
        self._current_step_prepared = -1

        # Store stats for external logging (e.g., TensorBoard)
        self.last_beta2_stats = {}

        # This ensures the map is complete before the first backward pass,
        # making it compatible with fused back pass mechanisms.
        self._build_layer_info_if_needed()

        if self.optimizer.param_groups[0].get('k_logging', 0) > 0:
            self.print_layer_info()

    def _build_layer_info_if_needed(self):
        """Builds a map of layers and the parameters they contain."""
        if self._layer_info_built:
            return

        if hasattr(self.optimizer, 'layer_key_fn') and self.optimizer.layer_key_fn is not None:
            # A custom key function was provided by the user. We will use it.
            pass
        else:
            # No key function was provided. Default to coarse, shape-based bucketing.
            self.optimizer.layer_key_fn = lambda p: \
                (id(p),) if p.dim() == 2 and 1 <= p.shape[0] <= 10 and p.shape[1] in {768, 1280, 4096} \
                else tuple(p.shape)
            # This ensures that we won't mix embeddings with tokens (1 to 10)
            # TODO find a better way to safeguard the embeddings

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

    def print_layer_info(self):
        """Prints the contents of self.layer_info for debugging."""
        print("\n--- BEGIN self.layer_info DUMP ---")
        if not self.layer_info:
            print("Layer info is empty. Make sure the optimizer has parameters.")
            return

        for layer_key, info in self.layer_info.items():
            param_count = len(info['params'])
            first_param_details = ""
            if param_count > 0:
                p = info['params'][0]
                first_param_details = f" (Example param shape: {list(p.shape)}, dtype: {p.dtype})"
            
            print(f"Key: {layer_key}, Params: {param_count}{first_param_details}")

        print("--- END self.layer_info DUMP ---\n")

    def prepare_step(self, current_step: int):
        """
        Calculates dynamic beta2 for all layers using the completed scalar accumulators
        from the PREVIOUS step. Should be called once at the start of an optimizer step.
        """
        
        beta2_log = []
        first_layer_key = next(iter(self.layer_info), None)
        # These are just for the sample log, initialize them
        sun, pooled_grad_norm, prev_r_ema_val, r_ema_tensor = (torch.tensor(0.0),)*4

        for layer_key, info in self.layer_info.items():
            params, group = info['params'], info['group_ref']

            first_param_in_layer = info['params'][0]
            param_state = self.optimizer.state[first_param_in_layer]

            if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                    'sum_sq_accumulator': torch.tensor(0.0, device=first_param_in_layer.device, dtype=torch.float32)
                }
            
            if 'kourkoutas_r_ema' not in param_state:
                param_state['kourkoutas_r_ema'] = torch.tensor(0.0, device=first_param_in_layer.device, dtype=torch.float32)

            r_ema_tensor = param_state['kourkoutas_r_ema']
            accumulator = self.layer_state[layer_key]['sum_sq_accumulator']
            
            pooled_grad_norm = torch.sqrt(accumulator)
            prev_r_ema_val = r_ema_tensor.item() # for logging
            
            # Update the persistent EMA tensor in-place.
            r_ema_tensor.mul_(group['ema_alpha']).add_(pooled_grad_norm, alpha=1.0 - group['ema_alpha'])
            
            beta2_max = group['betas'][1]
            sun = torch.tensor(0.0, device=r_ema_tensor.device) # Default sun to 0 for warmup
            
            if current_step < group['k_warmup_steps']:
                beta2 = beta2_max
            else:
                raw = pooled_grad_norm / (r_ema_tensor + group['tiny_spike'])
                sun = raw / (1.0 + raw)
                beta2 = beta2_max - (beta2_max - group['beta2_min']) * sun

            # Store the final calculated beta2 in the helper's transient state for this step.
            self.layer_state[layer_key]['dynamic_beta2'] = beta2.item() if isinstance(beta2, torch.Tensor) else beta2
            
            # Reset the accumulator for the next optimizer step.
            accumulator.zero_()

            beta2_log.append(self.layer_state[layer_key]['dynamic_beta2'])

        # Always compute stats for TensorBoard
        if beta2_log:
            beta2_tensor = torch.tensor(beta2_log, device='cpu')
            self.last_beta2_stats = {
                'min': beta2_tensor.min().item(),
                'max': beta2_tensor.max().item(),
                'mean': beta2_tensor.mean().item(),
            }

        # Handle periodic console logging
        k_logging_interval = self.optimizer.param_groups[0].get('k_logging', 0)
        is_logging_step = k_logging_interval > 0 and (current_step + 1) % k_logging_interval == 0
        if is_logging_step and self.last_beta2_stats:
            if first_layer_key:
                print(f"\n[Kourkoutas-β Debug] Step {current_step + 1} - Sample Layer '{first_layer_key}':")
                print(f"  - Grad Norm: {pooled_grad_norm.item():.4e}, Prev EMA: {prev_r_ema_val:.4e}, New EMA: {r_ema_tensor.item():.4e}")
                print(f"  - Sunspike: {sun.item():.4f}, Dynamic Beta2: {self.layer_state[first_layer_key]['dynamic_beta2']:.4f}")
            print(f"[Kourkoutas-β Debug] Step {current_step + 1} Overall Beta2 Stats: Min={self.last_beta2_stats['min']:.4f}, Max={self.last_beta2_stats['max']:.4f}, Mean={self.last_beta2_stats['mean']:.4f}")

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
            # Initialize the transient state for this layer if it's the first time in the step.
            if layer_key not in self.layer_state:
                    self.layer_state[layer_key] = {
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