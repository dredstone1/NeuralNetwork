#include <cuda_runtime.h>
#include "tensor_gpu.hpp"
#include <cstddef>
#include <stdexcept>

namespace tensor_gpu {

// Allocate memory on GPU for a tensor.
float* allocate(std::size_t count) {
    float* devicePtr = nullptr;
    cudaError_t err = cudaMalloc(&devicePtr, count * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    return devicePtr;
}

// Free GPU memory.
void deallocate(float* devicePtr) {
    if (devicePtr) {
        cudaFree(devicePtr);
    }
}

// Copy data from CPU to GPU.
void copyToDevice(float* deviceDst, const float* hostSrc, std::size_t count) {
    cudaMemcpy(deviceDst, hostSrc, count * sizeof(float), cudaMemcpyHostToDevice);
}

// Copy data from GPU to CPU.
void copyToHost(float* hostDst, const float* deviceSrc, std::size_t count) {
    cudaMemcpy(hostDst, deviceSrc, count * sizeof(float), cudaMemcpyDeviceToHost);
}

// Kernel to set all elements to zero.
__global__ void zeroKernel(float* data, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = 0.0f;
    }
}

// Set all elements to zero (on GPU).
void zero(float* deviceData, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    zeroKernel<<<numBlocks, blockSize>>>(deviceData, count);
    cudaDeviceSynchronize();
}

// Kernel for element-wise addition: C = A + B
__global__ void addKernel(const float* A, const float* B, float* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}

// Element-wise addition: C = A + B
void add(const float* A, const float* B, float* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    cudaDeviceSynchronize();
}

// Kernel for element-wise multiplication: C = A * B
__global__ void multiplyKernel(const float* A, const float* B, float* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] * B[idx];
    }
}

// Element-wise multiply: C = A * B
void multiply(const float* A, const float* B, float* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    cudaDeviceSynchronize();
}

// Dot product kernel using parallel reduction (simplified version)
__global__ void dotKernel(const float* A, const float* B, float* partialSum, std::size_t count) {
    __shared__ float cache[256];
    std::size_t tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + tid;

    float temp = 0.0f;
    if (idx < count) {
        temp = A[idx] * B[idx];
    }
    cache[tid] = temp;
    __syncthreads();

    // Reduction in shared memory
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSum[blockIdx.x] = cache[0];
    }
}

// Dot product between two vectors (A · B)
float dot(const float* A, const float* B, std::size_t count) {
    const std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;

    // Allocate partial sums
    float* d_partialSum = nullptr;
    cudaMalloc(&d_partialSum, numBlocks * sizeof(float));

    dotKernel<<<numBlocks, blockSize>>>(A, B, d_partialSum, count);
    cudaDeviceSynchronize();

    // Copy partial sums to host
    float* h_partialSum = new float[numBlocks];
    cudaMemcpy(h_partialSum, d_partialSum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float totalSum = 0.0f;
    for (std::size_t i = 0; i < numBlocks; i++) {
        totalSum += h_partialSum[i];
    }

    delete[] h_partialSum;
    cudaFree(d_partialSum);
    return totalSum;
}

// Kernel to apply ReLU activation: max(0, x)
__global__ void reluKernel(float* data, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

// Apply activation function (e.g., ReLU)
void relu(float* deviceData, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(deviceData, count);
    cudaDeviceSynchronize();
}

// Kernel to apply ReLU derivative:
// output[i] = input[i] > 0 ? 1 : 0
__global__ void reluDerivativeKernel(const float* input, float* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Apply derivative of activation function (e.g., ReLU')
void relu_derivative(const float* input, float* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
    cudaDeviceSynchronize();
}
} // namespace tensor_gpu
