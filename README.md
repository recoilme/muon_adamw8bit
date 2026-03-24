# muon_adamw8bit
A hybrid PyTorch optimizer that combines Muon (for matrix parameters/weights) and AdamW8bit (for scalar parameters/biases/norms).

# Installation
pip install git+https://github.com/recoilme/muon_adamw8bit.git

# Usage
```
import torch
from muon_adamw8bit import MuonAdamW8bit

model = MyModel()

# Initialize optimizer
# Muon requires higher LR. If base_lr=1e-5, muon_lr_mult=1000 results in 0.01 for Muon.
optimizer = MuonAdamW8bit(
    model.parameters(),
    lr=1e-5,
    muon_lr_mult=1000.0,       # Default: 1000.0
    ns_dtype=torch.bfloat16    # Default: torch.bfloat16. Use None for input dtype.
)

# Standard training loop
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
# Features

 - Automatic Parameter Splitting: Automatically detects ndim >= 2 (Matrices) for Muon and ndim < 2 (Scalars) for AdamW8bit.
 - Memory Efficient: Uses 8-bit quantization for Adam states via bitsandbytes.
 - Scheduler Compatible: Correctly synchronizes Learning Rate changes from schedulers to internal optimizers.
 - Conv2d Support: Correctly flattens Convolutional layers for orthogonalization.