// [REFERENCE_START]
//
// Simple row-major GEMM: C[M,N] = A[M,K] x B[K,N]
//
// Each work-item computes one element of C.
// Global work size: (N, M)  [id(0)=col, id(1)=row]
//
__kernel void gemm(__global const float* A,
                       __global const float* B,
                       __global float* C,
                       const int M,
                       const int K,
                       const int N)
{
    const int col = get_global_id(0); // output column in [0, N)
    const int row = get_global_id(1); // output row    in [0, M)

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;
}
// [REFERENCE_END]
