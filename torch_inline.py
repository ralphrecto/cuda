# %%
import torch
from torch.utils import cpp_extension

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda")


# %%
def load_kernel_module(name: str):
    module = cpp_extension.load(
        name=name,
        sources=[
            f"src/torch_kernels/{name}/main.cpp",
            f"src/torch_kernels/{name}/main.cu",
        ],
        extra_cflags=['-O2'],  # Reduce from -O3 to -O2 for faster compile
        extra_cuda_cflags=[
            '-O2',  # Reduce optimization during development
            '--use_fast_math',  # Enable fast math
            '-gencode=arch=compute_75,code=sm_75',  # Only compile for T4 GPU (MUCH faster!)
        ],
        verbose=True,
        with_cuda=True,
    )
    return module



# %%
module = load_kernel_module("example")

# %%
x = torch.randn(10000, 10000, device="cuda", dtype=torch.float32)
y = torch.randn(10000, 10000, device="cuda", dtype=torch.float32)

#%%
z = module.add(x, y).cpu()
# %%
%%timeit -n 10 -r 1000
module.sum_fast(torch.rand(100000, device="cuda", dtype=torch.float32))

#%%
%%timeit -n 10 -r 1000
torch.sum(torch.rand(100000, dtype=torch.float32))

#%%
module.sum_fast(torch.rand(100000, device="cuda", dtype=torch.float32))

#%%
module.sum(torch.rand(100000, device="cuda", dtype=torch.float32))

