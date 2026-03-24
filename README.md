# muon_adamw8bit
A hybrid PyTorch optimizer that combines Muon (for matrix parameters/weights) and AdamW8bit (for scalar parameters/biases/norms).

# Installation
```
pip install git+https://github.com/recoilme/muon_adamw8bit.git
```

# Usage
```
from muon_optimizer import MuonAdamW8bit

# ... define model ...

# Example 1: Basic usage
optimizer = MuonAdamW8bit(
    vae.parameters(), 
    lr=6e-6, 
    muon_lr_mult=1000,     # Muon LR will be 6e-6 * 1000 = 0.006
    ns_dtype=torch.bfloat16 # Default, safe and fast. Use None for input dtype.
)

# Example 2: With parameter groups (e.g., weight decay)
optimizer = MuonAdamW8bit(
    param_groups,           # list of dicts [{'params': ...}, ...]
    lr=6e-6,
    weight_decay=0.01,
    muon_lr_mult=1500
)

# Training loop
for batch in dataloader:
    loss = compute_loss(...)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
# Features

 - Automatic Parameter Splitting: Automatically detects ndim >= 2 (Matrices) for Muon and ndim < 2 (Scalars) for AdamW8bit.
 - Memory Efficient: Uses 8-bit quantization for Adam states via bitsandbytes.
 - Scheduler Compatible: Correctly synchronizes Learning Rate changes from schedulers to internal optimizers.
 - Conv2d Support: Correctly flattens Convolutional layers for orthogonalization.