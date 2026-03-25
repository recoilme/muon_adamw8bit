import torch
import bitsandbytes as bnb
import torch.distributed as dist

def zeropower_via_newtonschulz5(
    G: torch.Tensor, 
    steps: int = 5, 
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Newton-Schulz iteration. 
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    if G.dtype == torch.float16:
        X = G.float()
    else:
        X = G

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    return X.mT if transposed else X


class MuonInternal(torch.optim.Optimizer):
    """
    Internal Muon optimizer logic.
    """
    def __init__(
        self, 
        params, 
        lr: float = 0.01, 
        momentum: float = 0.95, 
        ns_steps: int = 5
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.distributed = dist.is_initialized()

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=torch.float32)
                
                buf = state["momentum_buffer"]
                
                buf.mul_(momentum).add_(g.float())
                
                update = g.float().add(buf, alpha=momentum)
                
                original_shape = g.shape
                needs_reshaping = g.ndim > 2
                
                if needs_reshaping:
                    update = update.reshape(update.size(0), -1)
                
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
                
                if needs_reshaping:
                    update = update.reshape(original_shape)
                
                p.add_(update, alpha=-lr)
                
        return loss


class MuonAdamW8bit(torch.optim.Optimizer):
    """
    Hybrid Optimizer: Muon + AdamW8bit.
    KISS Fix: Muon использует фиксированный LR, Adam использует LR от шедулера.
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        muon_lr_mult: float = 500.0,
        muon_momentum: float = 0.95, 
        ns_steps: int = 5
    ):
        
        params_list = list(params)
        param_groups_input = []

        if len(params_list) > 0:
            if isinstance(params_list[0], dict):
                param_groups_input = params_list
            else:
                param_groups_input = [{'params': params_list}]
        
        muon_groups = []
        adam_groups = []
        
        for group in param_groups_input:
            group_params = group['params']
            group_args = {k: v for k, v in group.items() if k != 'params'}
            
            matrix_ps = []
            scalar_ps = []
            
            for p in group_params:
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

        # Фиксируем LR для Muon один раз при инициализации
        actual_muon_lr = lr * muon_lr_mult
        self.muon_lr_scale = muon_lr_mult
        self.ns_steps = ns_steps

        self.muon_opt = MuonInternal(
            muon_groups, 
            lr=actual_muon_lr, 
            momentum=muon_momentum, 
            ns_steps=ns_steps
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
        
        # Шедулер будет дергать param_groupsAdam, а Muon будет жить своей жизнью
        if len(self.adam_opt.param_groups) > 0:
            self.param_groups = self.adam_opt.param_groups
        elif len(self.muon_opt.param_groups) > 0:
            self.param_groups = self.muon_opt.param_groups
        else:
            self.param_groups = []

    def zero_grad(self, set_to_none=False):
        self.muon_opt.zero_grad(set_to_none)
        self.adam_opt.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # --- KISS решение ---
        # Мы УБРАЛИ синхронизацию LR.
        # Muon продолжает использовать LR, заданный в __init__ (actual_muon_lr).
        # Adam использует LR, который меняет шедулер (с прогревом).
        # Это безопасно, так как Muon устойчив к высокому LR.
        
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
        
        # Восстанавливаем ссылки
        if len(self.adam_opt.param_groups) > 0:
            self.param_groups = self.adam_opt.param_groups
        elif len(self.muon_opt.param_groups) > 0:
            self.param_groups = self.muon_opt.param_groups