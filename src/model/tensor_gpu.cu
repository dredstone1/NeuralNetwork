#include <cuda_runtime.h>
#include "tensor_gpu.hpp"
#include <cstddef>
#include <stdexcept>

namespace nn::global::tensor_gpu {

// ==================================================
// Memory Management
// ==================================================
void* allocate(std::size_t size) {
    void* devicePtr = nullptr;
    if (cudaMalloc(&devicePtr, size) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    return devicePtr;
}

void deallocate(void* devicePtr) {
    if (devicePtr) {
        cudaFree(devicePtr);
    }
}

void copyToDevice(void* deviceDst, const void* hostSrc, std::size_t size) {
    cudaMemcpy(deviceDst, hostSrc, size, cudaMemcpyHostToDevice);
}

void copyDeviceToDevice(void* deviceDst, const void* deviceSrc, std::size_t size) {
    cudaMemcpy(deviceDst, deviceSrc, size, cudaMemcpyDeviceToDevice);
}

void copyToHost(void* hostDst, const void* deviceSrc, std::size_t size) {
    cudaMemcpy(hostDst, deviceSrc, size, cudaMemcpyDeviceToHost);
}

void setValueAt(ValueType* devicePtr, std::size_t index, ValueType value) {
    cudaMemcpy(devicePtr + index, &value, sizeof(ValueType), cudaMemcpyHostToDevice);
}

ValueType getValueAt(const ValueType* devicePtr, std::size_t index) {
    ValueType value;
    cudaMemcpy(&value, devicePtr + index, sizeof(ValueType), cudaMemcpyDeviceToHost);
    return value;
}

// ==================================================
// Utility Kernels
// ==================================================
__global__ void zeroKernel(ValueType* data, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) data[idx] = 0.0f;
}

void zero(ValueType* deviceData, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    zeroKernel<<<numBlocks, blockSize>>>(deviceData, count);
}

// ==================================================
// Vector-Vector Operations
// ==================================================
__global__ void addVecKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] + B[idx];
}

__global__ void subVecKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] - B[idx];
}

__global__ void mulVecKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] * B[idx];
}

__global__ void divVecKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] / B[idx];
}

void add_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addVecKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void subtraction_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    subVecKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void multiply_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    mulVecKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void division_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divVecKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// ==================================================
// Vector-Scalar Operations
// ==================================================
__global__ void addScalarKernel(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] + B;
}

__global__ void subScalarKernel(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] - B;
}

__global__ void mulScalarKernel(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] * B;
}

__global__ void divScalarKernel(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) C[idx] = A[idx] / B;
}

void add_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void subtraction_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    subScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void multiply_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    mulScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

void division_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// ==================================================
// Activation Functions
// ==================================================
__global__ void reluKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
}

__global__ void reluDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
}

void relu(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(input, output, count);
}

void relu_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

__global__ void sigmoidKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void sigmoidDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        ValueType s = 1.0f / (1.0f + expf(-x));
        output[idx] = s * (1.0f - s);
    }
}

void sigmoid(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(input, output, count);
}

void sigmoid_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

__global__ void tanhKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = tanhf(input[idx]);
}

__global__ void tanhDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType t = tanhf(input[idx]);
        output[idx] = 1.0f - t * t;
    }
}

void tanh_activation(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhKernel<<<numBlocks, blockSize>>>(input, output, count);
}

void tanh_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

__global__ void leakyReluKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = (input[idx] > 0.0f) ? input[idx] : alpha * input[idx];
}

__global__ void leakyReluDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = (input[idx] > 0.0f) ? 1.0f : alpha;
}

void leaky_relu(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
}

void leaky_relu_derivative(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
}

// ==================================================
// Softmax
// ==================================================
__global__ void softmaxKernel(const ValueType* input, ValueType* output, std::size_t count) {
    extern __shared__ ValueType shared[];

    std::size_t tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + tid;
    if (idx >= count) return;

    shared[tid] = (idx < count) ? input[idx] : -INFINITY;
    __syncthreads();

    ValueType max_val = shared[0];
    for (std::size_t i = 1; i < blockDim.x && (blockIdx.x * blockDim.x + i) < count; ++i) {
        max_val = fmaxf(max_val, shared[i]);
    }
    __syncthreads();

    ValueType e = expf(shared[tid] - max_val);
    shared[tid] = e;
    __syncthreads();

    ValueType sum = 0.0f;
    for (std::size_t i = 0; i < blockDim.x && (blockIdx.x * blockDim.x + i) < count; ++i) {
        sum += shared[i];
    }
    __syncthreads();

    output[idx] = shared[tid] / sum;
}

void softmax(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    std::size_t sharedMemSize = blockSize * sizeof(ValueType);
    softmaxKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, output, count);
}

// ==================================================
// Index Utilities
// ==================================================
__global__ void flattenIndexKernel(const size_t* indices, const size_t* shape,
                                   const size_t* strides, size_t ndim, size_t* outIndex) {
    size_t idx = 0;
    for (size_t i = 0; i < ndim; ++i) {
        if (indices[i] >= shape[i]) {
            *outIndex = size_t(-1);
            return;
        }
        idx += indices[i] * strides[i];
    }
    *outIndex = idx;
}

__global__ void computeFlatIndexKernel(const size_t* indices, const size_t* strides,
                                       size_t rank, size_t* outIndex) {
    size_t flatIndex = 0;
    for (size_t i = 0; i < rank; ++i) {
        flatIndex += indices[i] * strides[i];
    }
    *outIndex = flatIndex;
}

// ==================================================
// Matrix Operations
// ==================================================
__global__ void matmulKernel(const ValueType* A, const ValueType* B, ValueType* R, size_t M, size_t K) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        ValueType sum = 0;
        for (size_t j = 0; j < K; ++j) {
            sum += A[row * K + j] * B[j];
        }
        R[row] = sum;
    }
}

__global__ void outerKernel(const ValueType* a, const ValueType* b, ValueType* result, size_t m, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = m * n;
    if (idx < total) {
        size_t i = idx / n;
        size_t j = idx % n;
        result[i * n + j] = a[i] * b[j];
    }
}

__global__ void matmulTKernel(const ValueType* W, const ValueType* V, ValueType* R, size_t M, size_t N) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        ValueType sum = 0.0f;
        for (size_t i = 0; i < M; ++i) {
            sum += W[i * N + col] * V[i];
        }
        R[col] = sum;
    }
}

void matmul(const ValueType* A, const ValueType* B, ValueType* R, size_t M, size_t K) {
    const int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    matmulKernel<<<gridSize, blockSize>>>(A, B, R, M, K);
    cudaDeviceSynchronize();
}

void outer(const ValueType* a, const ValueType* b, ValueType* result, size_t m, size_t n) {
    const int blockSize = 256;
    int gridSize = (m * n + blockSize - 1) / blockSize;
    outerKernel<<<gridSize, blockSize>>>(a, b, result, m, n);
    cudaDeviceSynchronize();
}

void matmulT(const ValueType* W, const ValueType* V, ValueType* R, size_t M, size_t N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    matmulTKernel<<<gridSize, blockSize>>>(W, V, R, M, N);
    cudaDeviceSynchronize();
}

} // namespace nn::global::tensor_gpu
