#%%
import torch
from torch.utils.cpp_extension import load_inline

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda")

#%%
def load_kernel_module(name: str):
    module = torch.utils.cpp_extension.load(
        name=name,
        sources=[
            f"src/torch_kernels/{name}/main.cpp",
            f"src/torch_kernels/{name}/main.cu"],
        verbose=True,
    )
    return module

#%%
module = load_kernel_module("example")

#%%
x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

z = module.sigmoid_add(x, y).cpu()
z