import torch
import bitsandbytes as bnb
from typing import Optional, Union, List, Dict, Any

def zeropower_via_newtonschulz5(
    G: torch.Tensor, 
    steps: int = 10, 
    eps: float = 1e-7, 
    dtype: Optional[torch.dtype] = torch.bfloat16
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Supports batching and different precision modes.

    Args:
        G (torch.Tensor): Gradient tensor (can be 2D or 4D for Conv2d).
        steps (int): Number of Newton-Schulz iterations. Default: 10.
        eps (float): Epsilon for numerical stability. Default: 1e-7.
        dtype (Optional[torch.dtype]): Data type for computation. 
            - torch.bfloat16 (default): Fast and stable on modern GPUs.
            - None: Uses the input tensor's dtype (e.g., FP32).
    
    Returns:
        torch.Tensor: Orthogonalized update matrix.
    
    Example:
        >>> grad = torch.randn(64, 64)
        >>> update = zeropower_via_newtonschulz5(grad, steps=5, dtype=torch.bfloat16)
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Precision handling
    if dtype is not None:
        X = G.to(dtype)
    else:
        X = G

    # Transpose logic for efficiency (M > N)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    
    # Normalization
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    
    # Iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    return X.mT if transposed else X


class MuonInternal(torch.optim.Optimizer):
    """
    Internal Muon optimizer logic for matrix parameters (weights).
    """
    def __init__(
        self, 
        params, 
        lr: float = 0.01, 
        momentum: float = 0.95, 
        ns_steps: int = 5, 
        weight_decay: float = 0.0,
        ns_dtype: Optional[torch.dtype] = torch.bfloat16
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay, ns_dtype=ns_dtype)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group.get("weight_decay", 0.0)
            ns_dtype = group.get("ns_dtype", torch.bfloat16)

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)
                
                buf = state["momentum_buffer"]
                
                # Weight Decay (Decoupled)
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)
                
                # Momentum update
                buf.mul_(momentum).add_(g.float())
                
                # Nesterov lookahead
                update = g.float().add(buf, alpha=momentum)
                
                # Conv2d flattening
                original_shape = g.shape
                if g.ndim == 4:
                    update = update.view(update.size(0), -1)
                
                # Orthogonalization
                update = zeropower_via_newtonschulz5(update, steps=ns_steps, dtype=ns_dtype)
                
                # Scaling
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
                
                # Restore shape
                if g.ndim == 4:
                    update = update.view(original_shape)
                
                p.add_(update, alpha=-lr)
                
        return loss


class MuonAdamW8bit(torch.optim.Optimizer):
    """
    Hybrid Optimizer: Muon for matrices (weights) + AdamW8bit for scalars (biases, norms).
    
    Automatically separates parameters based on dimensionality.
    
    Args:
        params: Model parameters (generator, list of tensors, or list of param groups).
        lr (float): Base learning rate. Default: 1e-3.
        muon_lr_mult (float): Multiplier for Muon LR. Default: 1000.0.
        ns_dtype (Optional[torch.dtype]): Precision for Newton-Schulz. Default: torch.bfloat16.
    
    Example:
        >>> opt = MuonAdamW8bit(model.parameters(), lr=1e-5)
    """
    def __init__(
        self, 
        params, 
        lr: float = 4e-5, 
        betas: tuple = (0.9, 0.995), 
        eps: float = 1e-7, 
        weight_decay: float = 0.01, 
        muon_lr_mult: float = 1000.0,
        muon_momentum: float = 0.95, 
        ns_steps: int = 5,
        ns_dtype: Optional[torch.dtype] = torch.bfloat16
    ):
        
        # Calculate actual Muon LR
        actual_muon_lr = lr * muon_lr_mult
        self.muon_lr_scale = muon_lr_mult
        self.ns_steps = ns_steps
        self.ns_dtype = ns_dtype
        
        # --- ROBUST INPUT HANDLING ---
        # Convert generator/iterator to list to handle it safely
        params_list = list(params)
        
        param_groups_input = []

        if len(params_list) > 0:
            # Check if it's a list of dictionaries (param groups)
            # We check the first element.
            if isinstance(params_list[0], dict):
                # It looks like param groups. Validate them.
                for i, group in enumerate(params_list):
                    if not isinstance(group, dict):
                         raise ValueError(f"Parameter at index {i} is a dict, but element at {i} is not. Mixed input is not supported.")
                    if 'params' not in group:
                         raise ValueError(f"Parameter group at index {i} is missing 'params' key. Check your input.")
                param_groups_input = params_list
            else:
                # It's a list of tensors (parameters). Wrap it.
                param_groups_input = [{'params': params_list}]
        
        # Split parameters into Muon (matrix) and Adam (scalar) groups
        muon_groups = []
        adam_groups = []
        
        for group in param_groups_input:
            group_params = group['params']
            group_args = {k: v for k, v in group.items() if k != 'params'}
            
            matrix_ps = []
            scalar_ps = []
            
            for p in group_params:
                # Filter non-tensors just in case
                if not isinstance(p, torch.Tensor):
                    continue
                    
                if p.ndim >= 2:
                    matrix_ps.append(p)
                else:
                    scalar_ps.append(p)
            
            if matrix_ps:
                muon_groups.append({'params': matrix_ps, **group_args})
            if scalar_ps:
                adam_groups.append({'params': scalar_ps, **group_args})

        # Initialize internal optimizers
        self.muon_opt = MuonInternal(
            muon_groups, 
            lr=actual_muon_lr, 
            momentum=muon_momentum, 
            ns_steps=ns_steps, 
            weight_decay=weight_decay,
            ns_dtype=ns_dtype
        )
        
        self.adam_opt = bnb.optim.AdamW8bit(
            adam_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay
        )

        self.defaults = dict(lr=lr)
        self.state = {}
        
        # For Accelerate/LR Scheduler compatibility
        self.param_groups = self.adam_opt.param_groups

    def zero_grad(self, set_to_none=False):
        self.muon_opt.zero_grad(set_to_none)
        self.adam_opt.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # LR Synchronization
        if len(self.adam_opt.param_groups) > 0:
            current_base_lr = self.adam_opt.param_groups[0]['lr']
            for group in self.muon_opt.param_groups:
                group['lr'] = current_base_lr * self.muon_lr_scale
            
        self.muon_opt.step(closure)
        self.adam_opt.step(closure)

    def state_dict(self):
        return {
            'muon': self.muon_opt.state_dict(),
            'adam': self.adam_opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.muon_opt.load_state_dict(state_dict['muon'])
        self.adam_opt.load_state_dict(state_dict['adam'])