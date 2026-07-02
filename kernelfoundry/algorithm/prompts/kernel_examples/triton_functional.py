import torch
import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def forward(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE)
    return out
