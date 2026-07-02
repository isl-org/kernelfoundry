#include <torch/extension.h>
#include <cuda_runtime.h>

// Templated CUDA kernel
template <int BLOCK_X, int BLOCK_Y>
__global__ void elementwise_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int col = blockIdx.x * BLOCK_X + threadIdx.x;
    int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (row < N && col < M) {
        C[row * M + col] = A[row * M + col] * B[row * M + col];
    }
}

// Templated forward function
template <int BLOCK_X, int BLOCK_Y>
torch::Tensor forward_templated(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    auto C = torch::empty({N, M}, A.options());
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((M + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    elementwise_mul_kernel<BLOCK_X, BLOCK_Y><<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M
    );
    return C;
}

// Dispatch function - must have arguments corresponding to the template parameters of forward_templated
torch::Tensor forward(torch::Tensor A, torch::Tensor B, int block_x, int block_y) {
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

// Pybind11 interface
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Elementwise multiplication with block size dispatch");
}