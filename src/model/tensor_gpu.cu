#include <cuda_runtime.h>
#include <cstdio>
#include "tensor_gpu.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nn::global::tensor_gpu {

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    throw std::runtime_error(cudaGetErrorString(e)); \
  } \
} while(0)

// ==================================================
// Memory Management
// ==================================================
void* allocate(std::size_t size) {
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size));
    return devicePtr;
}

void deallocate(void* devicePtr) {
    if (devicePtr) {
        CUDA_CHECK(cudaFree(devicePtr));
    }
}

void copyToDevice(void* deviceDst, const void* hostSrc, std::size_t sizeBytes) {
    CUDA_CHECK(cudaMemcpy(deviceDst, hostSrc, sizeBytes, cudaMemcpyHostToDevice));
}

void copyDeviceToDevice(void* deviceDst, const void* deviceSrc, std::size_t sizeBytes) {
    CUDA_CHECK(cudaMemcpy(deviceDst, deviceSrc, sizeBytes, cudaMemcpyDeviceToDevice));
}

void copyToHost(void* hostDst, const void* deviceSrc, std::size_t sizeBytes) {
    CUDA_CHECK(cudaMemcpy(hostDst, deviceSrc, sizeBytes, cudaMemcpyDeviceToHost));
}

void setValueAt(ValueType* devicePtr, std::size_t index, ValueType value) {
    CUDA_CHECK(cudaMemcpy(devicePtr + index, &value, sizeof(ValueType), cudaMemcpyHostToDevice));
}

ValueType getValueAt(const ValueType* devicePtr, std::size_t index) {
    ValueType value;
    CUDA_CHECK(cudaMemcpy(&value, devicePtr + index, sizeof(ValueType), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
}

void division_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divVecKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
}

void subtraction_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    subScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    CUDA_CHECK(cudaGetLastError());
}

void multiply_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    mulScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    CUDA_CHECK(cudaGetLastError());
}

void division_scalar(const ValueType* A, ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divScalarKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    CUDA_CHECK(cudaGetLastError());
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
    if (idx < count) {
        ValueType derivative = (input[idx] > 0.0f) ? 1.0f : 0.0f;
        output[idx] *= derivative;
    }
}

void relu(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
}

void relu_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
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
        ValueType derivative = s * (1.0f - s);
        output[idx] *= derivative;
    }
}

void sigmoid(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void tanhKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = tanhf(input[idx]);
}

__global__ void tanhDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType t = tanhf(input[idx]);
        ValueType derivative = 1.0f - t * t;
        output[idx] *= derivative;
    }
}

void tanh_activation(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
}

void tanh_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void leakyReluKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = (input[idx] > 0.0f) ? input[idx] : alpha * input[idx];
}

__global__ void leakyReluDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType derivative = (input[idx] > 0.0f) ? 1.0f : alpha;
        output[idx] *= derivative;
    }
}

void leaky_relu(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_derivative(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
    CUDA_CHECK(cudaGetLastError());
}

// ==================================================
// Softmax
// ==================================================
__global__ void softmaxKernel(const ValueType* input, ValueType* output, std::size_t count) {
    extern __shared__ ValueType shared[];
    std::size_t tid = threadIdx.x;
    std::size_t blockStart = blockIdx.x * blockDim.x;
    std::size_t idx = blockStart + tid;

    // Load input into shared memory
    shared[tid] = (idx < count) ? input[idx] : -INFINITY;
    __syncthreads();

    // Compute max within this block
    ValueType max_val = shared[0];
    for (unsigned int i = 1; i < blockDim.x; ++i) {
        if (blockStart + i < count) {
            max_val = fmaxf(max_val, shared[i]);
        }
    }
    __syncthreads();

    // Compute exp(x - max) and store in shared memory
    ValueType e = (idx < count) ? expf(shared[tid] - max_val) : 0.0f;
    shared[tid] = e;
    __syncthreads();

    // Compute sum within this block
    ValueType sum = 0.0f;
    for (unsigned int i = 0; i < blockDim.x; ++i) {
        if (blockStart + i < count) {
            sum += shared[i];
        }
    }
    __syncthreads();

    // Normalize within this block
    if (idx < count) {
        output[idx] = shared[tid] / (sum == 0.0f ? 1.0f : sum);
    }
}

// Two-pass softmax for large vectors that need global normalization
__global__ void softmaxMaxKernel(const ValueType* input, ValueType* max_vals, std::size_t count) {
    extern __shared__ ValueType shared[];
    std::size_t tid = threadIdx.x;
    std::size_t blockStart = blockIdx.x * blockDim.x;
    std::size_t idx = blockStart + tid;

    // Load input into shared memory
    shared[tid] = (idx < count) ? input[idx] : -INFINITY;
    __syncthreads();

    // Compute max within this block
    ValueType max_val = shared[0];
    for (unsigned int i = 1; i < blockDim.x; ++i) {
        if (blockStart + i < count) {
            max_val = fmaxf(max_val, shared[i]);
        }
    }

    // Store block max
    if (tid == 0) {
        max_vals[blockIdx.x] = max_val;
    }
}

__global__ void softmaxSumKernel(const ValueType* input, const ValueType* max_vals,
                                ValueType* sums, std::size_t count) {
    extern __shared__ ValueType shared[];
    std::size_t tid = threadIdx.x;
    std::size_t blockStart = blockIdx.x * blockDim.x;
    std::size_t idx = blockStart + tid;

    // Load input and compute exp(x - global_max)
    ValueType x = (idx < count) ? input[idx] : 0.0f;
    ValueType e = (idx < count) ? expf(x - max_vals[0]) : 0.0f;
    shared[tid] = e;
    __syncthreads();

    // Compute sum within this block
    ValueType sum = 0.0f;
    for (unsigned int i = 0; i < blockDim.x; ++i) {
        if (blockStart + i < count) {
            sum += shared[i];
        }
    }

    // Store block sum
    if (tid == 0) {
        sums[blockIdx.x] = sum;
    }
}

__global__ void softmaxFinalKernel(const ValueType* input, const ValueType* max_vals,
                                  const ValueType* sums, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        ValueType e = expf(x - max_vals[0]);
        output[idx] = e / (sums[0] == 0.0f ? 1.0f : sums[0]);
    }
}

void softmax(const ValueType* input, ValueType* output, std::size_t count) {
    const std::size_t blockSize = 256;
    const std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    const std::size_t sharedMemSize = blockSize * sizeof(ValueType);

    // For small vectors, use single-block approach
    if (count <= blockSize) {
        softmaxKernel<<<1, count, count * sizeof(ValueType)>>>(input, output, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // For larger vectors, use three-pass approach
    // Allocate temporary buffers
    ValueType* d_max_vals = nullptr;
    ValueType* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_max_vals, numBlocks * sizeof(ValueType)));
    CUDA_CHECK(cudaMalloc(&d_sums, numBlocks * sizeof(ValueType)));

    // Pass 1: Find global maximum
    softmaxMaxKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, d_max_vals, count);
    CUDA_CHECK(cudaGetLastError());

    // Copy max values to host for final reduction
    std::vector<ValueType> h_max_vals(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_max_vals.data(), d_max_vals, numBlocks * sizeof(ValueType), cudaMemcpyDeviceToHost));

    // Find global max on host
    ValueType global_max = h_max_vals[0];
    for (std::size_t i = 1; i < numBlocks; ++i) {
        global_max = fmaxf(global_max, h_max_vals[i]);
    }

    // Copy global max back to device
    CUDA_CHECK(cudaMemcpy(d_max_vals, &global_max, sizeof(ValueType), cudaMemcpyHostToDevice));

    // Pass 2: Compute sums with global max
    softmaxSumKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, d_max_vals, d_sums, count);
    CUDA_CHECK(cudaGetLastError());

    // Copy sums to host for final reduction
    std::vector<ValueType> h_sums(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums, numBlocks * sizeof(ValueType), cudaMemcpyDeviceToHost));

    // Find global sum on host
    ValueType global_sum = 0.0f;
    for (std::size_t i = 0; i < numBlocks; ++i) {
        global_sum += h_sums[i];
    }

    // Copy global sum back to device
    CUDA_CHECK(cudaMemcpy(d_sums, &global_sum, sizeof(ValueType), cudaMemcpyHostToDevice));

    // Pass 3: Final normalization
    softmaxFinalKernel<<<numBlocks, blockSize>>>(input, d_max_vals, d_sums, output, count);
    CUDA_CHECK(cudaGetLastError());

    // Cleanup
    CUDA_CHECK(cudaFree(d_max_vals));
    CUDA_CHECK(cudaFree(d_sums));
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
        result[i * n + j] += a[i] * b[j];
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
    CUDA_CHECK(cudaGetLastError());
}

void outer(const ValueType* a, const ValueType* b, ValueType* result, size_t m, size_t n) {
    const int blockSize = 256;
    int gridSize = (m * n + blockSize - 1) / blockSize;
    outerKernel<<<gridSize, blockSize>>>(a, b, result, m, n);
    CUDA_CHECK(cudaGetLastError());
}

void matmulT(const ValueType* W, const ValueType* V, ValueType* R, size_t M, size_t N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    matmulTKernel<<<gridSize, blockSize>>>(W, V, R, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// ==================================================
// Max Index Reduction
// ==================================================
__device__ MaxIndex maxIndexOp(MaxIndex a, MaxIndex b) {
    return (a.value >= b.value) ? a : b;
}

__global__ void maxIndexReduceKernel(const ValueType* input, MaxIndex* blockResults, std::size_t count) {
    extern __shared__ MaxIndex sdata[];
    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * blockDim.x + tid;

    // Load input and index or default if out of range
    MaxIndex local;
    if (i < count) {
        local.value = input[i];
        local.index = i;
    } else {
        local.value = -INFINITY;
        local.index = size_t(-1);
    }
    sdata[tid] = local;
    __syncthreads();

    // Reduction in shared memory
    for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = maxIndexOp(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockResults[blockIdx.x] = sdata[0];
    }
}

std::size_t getMaxElementIndex(const ValueType* deviceData, std::size_t count) {
    if (count == 0) {
        throw std::runtime_error("getMaxElementIndex: empty array");
    }

    const std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;

    MaxIndex* d_blockResults = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockResults, numBlocks * sizeof(MaxIndex)));

    maxIndexReduceKernel<<<numBlocks, blockSize, blockSize * sizeof(MaxIndex)>>>(deviceData, d_blockResults, count);
    CUDA_CHECK(cudaGetLastError());

    // Copy block results to host
    std::vector<MaxIndex> h_blockResults(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_blockResults.data(), d_blockResults, numBlocks * sizeof(MaxIndex), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_blockResults));

    // Final reduction on host
    MaxIndex maxRes = h_blockResults[0];
    for (std::size_t i = 1; i < numBlocks; ++i) {
        if (h_blockResults[i].value > maxRes.value) {
            maxRes = h_blockResults[i];
        }
    }

    return maxRes.index;
}

__global__ void conv2dKernel(const ValueType* input, const ValueType* filters,
                             ValueType* output,
                             int H, int W, int F, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output row
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output col
    int f = blockIdx.z;                             // filter index

    if (x >= H - K + 1 || y >= W - K + 1) return;

    ValueType sum = 0.0f;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            sum += input[(x+i)*W + (y+j)] * filters[f*K*K + i*K + j];

    output[(f*(H-K+1) + x)*(W-K+1) + y] = sum;
}

void conv2d(const ValueType* input, const ValueType* filters, ValueType* output,
            int H, int W, int F, int K) {
    dim3 blockSize(16, 16); // each block computes 16x16 output pixels
    dim3 gridSize(
        (W - K + 1 + blockSize.x - 1) / blockSize.x,
        (H - K + 1 + blockSize.y - 1) / blockSize.y,
        F
    );

    conv2dKernel<<<gridSize, blockSize>>>(input, filters, output, H, W, F, K);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace nn::global::tensor_gpu
