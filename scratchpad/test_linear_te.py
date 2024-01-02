import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 4096
out_features = 4096
hidden_size = 4096

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

for _ in range(100):
    # Enable autocasting for the forward pass
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp)

    loss = out.sum()
    loss.backward()

print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
