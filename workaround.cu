#include <cuda_runtime.h>
#include <cstdio>

// 1. Empty kernel
__global__ void empty_kernel() {
    // no-op
}

// 2. Tiny memcpy kernel: one thread per element
__global__ void memcpy_kernel(const float* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

int main() {
    const size_t N = 1 << 20; // small buffer (~4 MB)
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    // Optionally initialize d_src to avoid uninitialized memory
    cudaMemset(d_src, 0, N * sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    for (int bit = 0; bit < 1000; ++bit) {
        if (bit % 2 == 0) {
            // HIGH freq bit: empty kernel
            empty_kernel<<<1,1>>>();
        } else {
            // LOW freq bit: memcpy kernel
            memcpy_kernel<<<grid, block>>>(d_src, d_dst, N);
        }
        cudaDeviceSynchronize();
        // insert timing or frequency query here if you want
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
