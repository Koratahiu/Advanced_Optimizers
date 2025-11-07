import torch
from torch.optim import Optimizer

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
            if not group.get('kourkoutas_beta', False) and not group.get('adam_kourkoutas_beta', False):
                continue

            for p in group['params']:
                # The mapping is static and should not depend on the presence of a gradient.
                layer_key = self.optimizer.layer_key_fn(p)
                if layer_key not in self.layer_info:
                    self.layer_info[layer_key] = {'params': [], 'group_ref': group}
                self.layer_info[layer_key]['params'].append(p)

        self._layer_info_built = True

    def prepare_step(self, current_step: int):
        """
        Calculates dynamic beta2 for all layers using the completed scalar accumulators
        from the PREVIOUS step. Should be called once at the start of an optimizer step.
        """

        beta2_log = []
        # These are just for the sample log, initialize them
        sun, pooled_grad_norm, r_ema_tensor = (torch.tensor(0.0),)*3

        # The optimizer that owns this helper holds the master defaults for K-b.
        # This is crucial in hybrid optimizers where some param_groups might not
        # have all K-b keys populated, preventing KeyErrors.
        master_defaults = self.optimizer.defaults

        for layer_key, info in self.layer_info.items():
            group = info['group_ref']

            if not group.get('kourkoutas_beta', False) and not group.get('adam_kourkoutas_beta', False):
                continue

            first_param_in_layer = info['params'][0]
            param_state = self.optimizer.state[first_param_in_layer]

            if layer_key not in self.layer_state:
                self.layer_state[layer_key] = {
                    'sum_sq_accumulator': torch.tensor(0.0, device=first_param_in_layer.device, dtype=torch.float32)
                }

            if 'kourkoutas_r_ema' not in param_state:
                param_state['kourkoutas_r_ema'] = torch.tensor(0.0, device=first_param_in_layer.device, dtype=torch.float32)

            # Use group-specific K-b settings, falling back to the optimizer's master defaults.
            # This makes the helper robust against param groups that enable kourkoutas_beta
            # but are missing the other required hyperparameters.
            # In hybrid optimizers like Muon_adv, the Kourkoutas-related keys in the
            # defaults and param_groups are prefixed with 'adam_' to avoid conflicts.
            # We must detect this case and use the correct key names.
            prefix = 'adam_' if group.get('adam_kourkoutas_beta', False) else ''

            ema_alpha = group.get(f'{prefix}ema_alpha', master_defaults[f'{prefix}ema_alpha'])
            betas_tuple = group.get(f'{prefix}betas', master_defaults[f'{prefix}betas'])
            beta2_max = betas_tuple[1]
            beta2_min = group.get(f'{prefix}beta2_min', master_defaults[f'{prefix}beta2_min'])
            tiny_spike = group.get(f'{prefix}tiny_spike', master_defaults[f'{prefix}tiny_spike'])
            k_warmup_steps = group.get(f'{prefix}k_warmup_steps', master_defaults[f'{prefix}k_warmup_steps'])

            r_ema_tensor = param_state['kourkoutas_r_ema']
            accumulator = self.layer_state[layer_key]['sum_sq_accumulator']

            pooled_grad_norm = torch.sqrt(accumulator)

            # Update the persistent EMA tensor in-place.
            r_ema_tensor.mul_(ema_alpha).add_(pooled_grad_norm, alpha=1.0 - ema_alpha)

            sun = torch.tensor(0.0, device=r_ema_tensor.device) # Default sun to 0 for warmup

            if current_step < k_warmup_steps:
                beta2 = beta2_max
            else:
                raw = pooled_grad_norm / (r_ema_tensor + tiny_spike)
                sun = raw / (1.0 + raw)
                beta2 = beta2_max - (beta2_max - beta2_min) * sun

            # Store the final calculated beta2 in the helper's transient state for this step.
            self.layer_state[layer_key]['dynamic_beta2'] = beta2.item() if isinstance(beta2, torch.Tensor) else beta2

            # Reset the accumulator for the next optimizer step.
            accumulator.zero_()

            beta2_log.append(self.layer_state[layer_key]['dynamic_beta2'])

        # Always compute stats for TensorBoard
        if beta2_log:
            beta2_tensor = torch.tensor(beta2_log, device='cpu')
            self.last_beta2_stats = {
                'mean': beta2_tensor.mean().item(),
            }

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

    def get_beta2(self, p: torch.Tensor, group: dict) -> float:
        """
        Gets the appropriate beta2 for the current parameter, handling warmup and dynamic value fetching.
        """
        layer_key = self.optimizer.layer_key_fn(p)
        # The default is the max value, which is correct for unmapped params or edge cases
        beta2_default = group.get('betas', group.get('adam_betas'))[1] if group.get('betas', group.get('adam_betas')) else 0.999
        return self.layer_state.get(layer_key, {}).get('dynamic_beta2', beta2_default)
