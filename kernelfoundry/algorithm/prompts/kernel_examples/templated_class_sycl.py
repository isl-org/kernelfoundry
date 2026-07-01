import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_multiply_sycl_source = """
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

// Kernel name struct at namespace scope
template <int BX, int BY>
struct ElementwiseMulKernel {};

template <int BLOCK_X, int BLOCK_Y>
void elementwise_mul_sycl_kernel(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int N,
    int M
) {
    auto a_data = A.data_ptr<float>();
    auto b_data = B.data_ptr<float>();
    auto c_data = C.data_ptr<float>();

    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

    sycl::range<2> global_range(
        ((N + BLOCK_Y - 1) / BLOCK_Y) * BLOCK_Y,
        ((M + BLOCK_X - 1) / BLOCK_X) * BLOCK_X
    );
    sycl::range<2> local_range(BLOCK_Y, BLOCK_X);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<ElementwiseMulKernel<BLOCK_X, BLOCK_Y>>(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                if (row < N && col < M) {
                    int idx = row * M + col;
                    c_data[idx] = a_data[idx] * b_data[idx];
                }
            }
        );
    }).wait();
}


// 2. Templated forward function
template <int BLOCK_X, int BLOCK_Y>
torch::Tensor forward_templated(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    auto C = torch::empty({N, M}, A.options());
    elementwise_mul_sycl_kernel<BLOCK_X, BLOCK_Y>(A, B, C, N, M);
    return C;
}

// 3. Dispatcher - must have arguments corresponding to the template parameters of forward_templated
torch::Tensor forward(torch::Tensor A, torch::Tensor B, int block_x, int block_y) {
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Only float32 supported in this example");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input sizes must match");

    if (block_x == 16 && block_y == 16) {
        return forward_templated<16, 16>(A, B);
    } else if (block_x == 32 && block_y == 8) {
        return forward_templated<32, 8>(A, B);
    } else if (block_x == 8 && block_y == 32) {
        return forward_templated<8, 32>(A, B);
    } else {
        TORCH_CHECK(false, "Unsupported block size combination");
    }
}

"""


elementwise_multiply_cpp_source = """
torch::Tensor forward(torch::Tensor a, torch::Tensor b, int block_x, int block_y);
"""


elementwise_multiply = load_inline(
    name="elementwise_multiply",
    cpp_sources=elementwise_multiply_cpp_source,
    sycl_sources=elementwise_multiply_sycl_source,
    functions=["forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_multiply = elementwise_multiply

    def forward(self, a, b, template_args):
        # template_args should be [block_x, block_y]
        return self.elementwise_multiply.forward(a, b, *template_args)
