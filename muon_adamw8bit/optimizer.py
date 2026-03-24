import torch
import bitsandbytes as bnb
from typing import Optional

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
            - If torch.bfloat16 (default): casts input to bf16 for speed.
            - If None: uses the input tensor's dtype.
    
    Returns:
        torch.Tensor: Orthogonalized update matrix in the original input dtype (if cast) or compute dtype.
    
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
        # Safety: FP16 is often unstable for NS iterations without careful scaling, 
        # but we respect the user choice if dtype=None and input is fp16.

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
    Implements momentum orthogonalization.
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
                
                # Weight Decay (Decoupled, similar to AdamW)
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
                
                # Cast back to param dtype if necessary (usually float32/bfloat16)
                p.add_(update, alpha=-lr)
                
        return loss


class MuonAdamW8bit(torch.optim.Optimizer):
    """
    Hybrid Optimizer: Muon for matrices (weights) + AdamW8bit for scalars (biases, norms).
    
    This optimizer automatically separates parameters based on their dimensionality.
    - Parameters with ndim >= 2 (Weights, Convs) are optimized using Muon.
    - Parameters with ndim < 2 (Biases, LayerNorms) are optimized using AdamW8bit.
    
    It handles Learning Rate scheduling correctly by synchronizing LR between Adam and Muon groups.
    
    Args:
        params: Model parameters (can be a generator or list of dicts).
        lr (float): Base learning rate for AdamW.
        betas (tuple): Betas for AdamW.
        eps (float): Epsilon for AdamW.
        weight_decay (float): Weight decay coefficient.
        muon_lr_mult (float): Multiplier for Muon LR relative to base LR. 
                              Muon typically requires higher LR (e.g., x1000).
                              Default: 1000.0.
        muon_momentum (float): Momentum for Muon. Default: 0.95.
        ns_steps (int): Newton-Schulz iterations. Default: 5.
        ns_dtype (torch.dtype): Precision for Newton-Schulz. Default: torch.bfloat16.
    
    Example:
        >>> model = MyModel()
        >>> opt = MuonAdamW8bit(model.parameters(), lr=1e-5, muon_lr_mult=1000)
        >>> loss.backward()
        >>> opt.step()
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-5, 
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
        
        # Split parameters
        muon_groups = []
        adam_groups = []
        
        params = list(params)
        
        for group in params:
            group_params = group['params']
            group_args = {k: v for k, v in group.items() if k != 'params'}
            
            matrix_ps = []
            scalar_ps = []
            
            for p in group_params:
                if p.ndim >= 2:
                    matrix_ps.append(p)
                else:
                    scalar_ps.append(p)
            
            if matrix_ps:
                muon_groups.append({'params': matrix_ps, **group_args})
            if scalar_ps:
                adam_groups.append({'params': scalar_ps, **group_args})

        # Initialize internal optimizers
        # Muon handles its own weight decay logic internally
        self.muon_opt = MuonInternal(
            muon_groups, 
            lr=actual_muon_lr, 
            momentum=muon_momentum, 
            ns_steps=ns_steps, 
            weight_decay=weight_decay,
            ns_dtype=ns_dtype
        )
        
        # AdamW8bit handles weight decay via bnb implementation
        self.adam_opt = bnb.optim.AdamW8bit(
            adam_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay
        )

        self.defaults = dict(lr=lr)
        self.state = {}
        
        # AdamW8bit handles weight decay via standard arguments
        self.adam_opt = bnb.optim.AdamW8bit(
            adam_groups, 
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay
        )

        self.defaults = dict(lr=lr)
        self.state = {}
        
        # For Accelerate/LR Scheduler compatibility, expose Adam's param_groups
        self.param_groups = self.adam_opt.param_groups

    def zero_grad(self, set_to_none=False):
        self.muon_opt.zero_grad(set_to_none)
        self.adam_opt.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # LR Synchronization
        # Accelerate updates LR in `optimizer.param_groups` (which is Adam's groups here).
        # We sync the updated LR to Muon groups applying the scale.
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