import torch
import torch.distributed as dist
from torch.optim import Optimizer

class ProdigyShapeHelper:
    """
    Helper class for Prodigy_adv to handle per-shape D-adaptation statistics.
    """
    def __init__(self, optimizer: Optimizer, d0: float, d_coef: float):
        self.optimizer = optimizer
        self.d0 = d0
        self.d_coef = d_coef

        # Map: shape_tuple -> master_parameter
        # We use the first parameter of a given shape to store the shared statistics.
        self.master_params = {}
        self._map_built = False

    def _build_map_if_needed(self):
        """Lazy initialization of the shape-to-parameter mapping."""
        if self._map_built:
            return

        for group in self.optimizer.param_groups:
            for p in group['params']:
                shape = tuple(p.shape)
                if shape not in self.master_params:
                    self.master_params[shape] = p

        self._map_built = True

    def get_state_for_shape(self, shape, device=None):
        """Retrieves or initializes the state dict for a given shape."""
        self._build_map_if_needed()
        if shape not in self.master_params:
            return None
            
        master_p = self.master_params[shape]
        state = self.optimizer.state[master_p]
        
        # Initialize keys if they don't exist (e.g. first run or after load_state_dict)
        if 'prodigy_d' not in state:
            dev = device if device is not None else master_p.device
            state['prodigy_d'] = self.d0
            state['prodigy_d_max'] = self.d0
            state['prodigy_numerator'] = torch.tensor(0.0, device=dev)
            state['prodigy_denom'] = torch.tensor(0.0, device=dev)
            
        return state

    def get_d(self, p: torch.Tensor) -> float:
        """Returns the d value for a specific parameter shape."""
        shape = tuple(p.shape)
        state = self.get_state_for_shape(shape, p.device)
        return state['prodigy_d']

    def get_shape_d_values(self) -> dict:
        """Returns a dictionary of {shape_tuple: d_value} for all buckets."""
        self._build_map_if_needed()
        d_values = {}
        for shape, master_p in self.master_params.items():
            state = self.optimizer.state[master_p]
            # Only include if initialized
            if 'prodigy_d' in state:
                shape_str = 'x'.join(map(str, shape))
                d_values[shape_str] = state['prodigy_d']
        return d_values

    def accumulate_stats(self, p: torch.Tensor, grad: torch.Tensor, 
                         p0: torch.Tensor, s: torch.Tensor,
                         d: float, dlr: float, beta3: float, 
                         safeguard_warmup: bool, slice_p: int):
        """
        Accumulates Prodigy statistics (numerator and s/denom) for a specific shape bucket.
        """
        shape = tuple(p.shape)
        state = self.get_state_for_shape(shape, p.device)
        
        # Access shared tensors
        num_tensor = state['prodigy_numerator']
        den_tensor = state['prodigy_denom']
        
        # Ensure stats are on the correct device
        num_tensor = num_tensor.to(p.device)
        den_tensor = den_tensor.to(p.device)
        state['prodigy_numerator'] = num_tensor
        state['prodigy_denom'] = den_tensor

        grad_slice = grad.flatten()[::slice_p].float()
        p_slice = p.flatten()[::slice_p].float()
        p0_slice = p0.float()

        # Numerator accumulation
        # nom += (d / d0) * dlr * <g, p0 - p>
        num_tensor.add_((d / self.d0) * dlr * torch.dot(grad_slice, p0_slice.data - p_slice))

        # Denominator (s) accumulation
        alpha = ((d / self.d0) * d) if safeguard_warmup else ((d / self.d0) * dlr)
        s.mul_(beta3).add_(grad_slice, alpha=alpha)

        # denom += ||s||_1
        den_tensor.add_(s.abs().sum())

    def compute_d_per_shape(self, growth_rate: float, d_limiter: bool, beta3: float):
        """
        Calculates new d values for all shape buckets.
        Handles distributed reduction if dist is initialized.
        """
        self._build_map_if_needed()
        if not self.master_params:
            return

        sorted_shapes = sorted(self.master_params.keys())

        # Prepare tensors for distributed reduction
        local_stats = []
        active_states = []

        for shape in sorted_shapes:
            master_p = self.master_params[shape]
            state = self.optimizer.state[master_p]
            # Only process initialized states
            if 'prodigy_numerator' in state:
                local_stats.append(state['prodigy_numerator'])
                local_stats.append(state['prodigy_denom'])
                active_states.append(state)

        if not local_stats:
            return

        packed_tensor = torch.stack(local_stats)

        # Distributed All-Reduce (Sum)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed_tensor, op=dist.ReduceOp.SUM)

        # Unpack and Calculate
        idx = 0
        for state in active_states:

            global_num = packed_tensor[idx].item()
            global_den = packed_tensor[idx+1].item()
            idx += 2

            # Apply Prodigy D-adaptation logic
            if global_den > 0:
                d_hat = self.d_coef * global_num / global_den

                cur_d = state['prodigy_d']
                cur_d_max = state['prodigy_d_max']

                if d_limiter:
                    d_hat = min(cur_d * (2 ** 0.25), d_hat)

                if cur_d == self.d0:
                    cur_d = max(cur_d, d_hat)

                cur_d_max = max(cur_d_max, d_hat)
                cur_d = min(cur_d_max, cur_d * growth_rate)

                state['prodigy_d'] = cur_d
                state['prodigy_d_max'] = cur_d_max

            # Prepare accumulators for next step
            state['prodigy_numerator'].mul_(beta3)
            state['prodigy_denom'].zero_()