#include <cuda_runtime.h>
#include "tensor_gpu.hpp"
#include <cstddef>
#include <stdexcept>

namespace nn::global::tensor_gpu {
// Allocate memory on GPU for a tensor.
void* allocate(std::size_t size) {
    void* devicePtr = nullptr;
    cudaError_t err1 = cudaMalloc(&devicePtr, size);
    if (err1 != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    return devicePtr;
}

// Free GPU memory.
void deallocate(void* devicePtr) {
    if (devicePtr) {
        cudaFree(devicePtr);
    }
}

// Copy data from CPU to GPU.
void copyToDevice(void* deviceDst, const void * hostSrc, std::size_t size) {
    cudaMemcpy(deviceDst, hostSrc, size, cudaMemcpyHostToDevice);
}


void copyDeviceToDevice(void *deviceDst, const void *deviceSrc, std::size_t size) {
    cudaMemcpy(deviceDst, deviceSrc, size, cudaMemcpyDeviceToDevice);
}

// Copy data from GPU to CPU.
void copyToHost(void* hostDst, const void* deviceSrc, std::size_t size) {
    cudaMemcpy(hostDst, deviceSrc, size, cudaMemcpyDeviceToHost);
}

// Kernel to set all elements to zero.
__global__ void zeroKernel(ValueType* data, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = 0.0f;
    }
}

// Set all elements to zero (on GPU).
void zero(ValueType* deviceData, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    zeroKernel<<<numBlocks, blockSize>>>(deviceData, count);
}

// Kernel for element-wise addition: C = A + B
__global__ void addKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}

// Element-wise addition: C = A + B
void add_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}


// Kernel for element-wise addition: C = A - B
__global__ void subtractionKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] - B[idx];
    }
}

// Element-wise addition: C = A + B
void subtraction_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    subtractionKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// Kernel for element-wise addition: C = A / B
__global__ void divisionKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] / B[idx];
    }
}

// Element-wise addition: C = A / B
void division_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divisionKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// Kernel for element-wise multiplication: C = A * B
__global__ void multiplyKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] * B[idx];
    }
}

// Element-wise multiply: C = A * B
void multiply_vec(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// Kernel for element-wise addition: C = A + B
__global__ void addKernel(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B;
    }
}

// Element-wise addition: C = A + B
void add_scalar(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}


// Kernel for element-wise addition: C = A - B
__global__ void subtractionKernel(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] - B;
    }
}

// Element-wise addition: C = A + B
void subtraction_scalar(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    subtractionKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// Kernel for element-wise addition: C = A / B
__global__ void divisionKernel(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] / B;
    }
}

// Element-wise addition: C = A / B
void division_scalar(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    divisionKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

// Kernel for element-wise multiplication: C = A * B
__global__ void multiplyKernel(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] * B;
    }
}

// Element-wise multiply: C = A * B
void multiply_scalar(const ValueType* A, const ValueType B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(A, B, C, count);
}

__global__ void computeStrides(const size_t *shape, size_t *strides, size_t ndim) {
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

void computeStridesDevice(const size_t *gpu_shape, size_t *gpu_strides, std::size_t ndim) {
    computeStrides<<<1, 1>>>(gpu_shape, gpu_strides, ndim);
    cudaDeviceSynchronize();
}

// Kernel to apply ReLU activation: max(0, x)
__global__ void reluKernel(const ValueType *input, ValueType *output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] > 0.0 ? input[idx] : 0.0f;
    }
}

// Apply activation function (e.g., ReLU)
void relu(const ValueType *input, ValueType *output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel to apply ReLU derivative:
// output[i] = input[i] > 0 ? 1 : 0
__global__ void reluDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Apply derivative of activation function (e.g., ReLU')
void relu_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    reluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel to apply Sigmoid activation: 1 / (1 + exp(-x))
__global__ void sigmoidKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// Apply Sigmoid activation
void sigmoid(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel for Sigmoid derivative: s(x) * (1 - s(x))
__global__ void sigmoidDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        ValueType s = 1.0f / (1.0f + expf(-x));
        output[idx] = s * (1.0f - s);
    }
}

// Apply Sigmoid derivative
void sigmoid_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    sigmoidDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel to apply Tanh activation: tanh(x)
__global__ void tanhKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = tanhf(input[idx]);
    }
}

// Apply Tanh activation
void tanh_activation(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel for Tanh derivative: 1 - tanh(x)^2
__global__ void tanhDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType t = tanhf(input[idx]);
        output[idx] = 1.0f - t * t;
    }
}

// Apply Tanh derivative
void tanh_derivative(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    tanhDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count);
}

// Kernel for Leaky ReLU: x > 0 ? x : alpha * x
__global__ void leakyReluKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ValueType x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * x;
    }
}

// Apply Leaky ReLU
void leaky_relu(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
}

// Kernel for Leaky ReLU derivative: x > 0 ? 1 : alpha
__global__ void leakyReluDerivativeKernel(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : alpha;
    }
}

// Apply Leaky ReLU derivative
void leaky_relu_derivative(const ValueType* input, ValueType* output, std::size_t count, ValueType alpha) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    leakyReluDerivativeKernel<<<numBlocks, blockSize>>>(input, output, count, alpha);
}

__global__ void softmaxKernel(const ValueType* input, ValueType* output, std::size_t count) {
    extern __shared__ ValueType shared[];

    std::size_t tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx >= count) return;

    // Load input into shared memory
    if (idx < count) shared[tid] = input[idx];
    else shared[tid] = -INFINITY; // or 0

    __syncthreads();

    // Step 1: Find max value for numerical stability
    ValueType max_val = shared[0];
    for (std::size_t i = 1; i < blockDim.x && blockIdx.x * blockDim.x + i < count; ++i) {
        max_val = fmaxf(max_val, shared[i]);
    }
    __syncthreads();

    // Step 2: Compute exp(x - max)
    ValueType e = expf(shared[tid] - max_val);
    shared[tid] = e;
    __syncthreads();

    // Step 3: Sum of exponentials
    ValueType sum = 0.0f;
    for (std::size_t i = 0; i < blockDim.x && blockIdx.x * blockDim.x + i < count; ++i) {
        sum += shared[i];
    }
    __syncthreads();

    // Step 4: Normalize
    output[idx] = shared[tid] / sum;
}

void softmax(const ValueType* input, ValueType* output, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    std::size_t sharedMemSize = blockSize * sizeof(ValueType);

    softmaxKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, output, count);
}

void setValueAt(ValueType* devicePtr, std::size_t index, ValueType value) {
    cudaMemcpy(devicePtr + index, &value, sizeof(ValueType), cudaMemcpyHostToDevice);
}

ValueType getValueAt(const ValueType* devicePtr , std::size_t index) {
    ValueType value;
    cudaMemcpy(&value, devicePtr + index, sizeof(ValueType), cudaMemcpyDeviceToHost);
    return value;
}

// Kernel to compute flattened index
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

// Host function to launch kernel
size_t flattenIndexGpu(const size_t* h_indices, const size_t* d_shape,
                       const size_t* d_strides, size_t ndim) {
    size_t *d_indices,  *d_outIndex;
    cudaMalloc(&d_indices, ndim * sizeof(size_t));
    cudaMalloc(&d_outIndex, sizeof(size_t));

    cudaMemcpy(d_indices, h_indices, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    flattenIndexKernel<<<1, 1>>>(d_indices, d_shape, d_strides, ndim, d_outIndex);
    cudaDeviceSynchronize();

    size_t result;
    cudaMemcpy(&result, d_outIndex, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_outIndex);

    if (result == size_t(-1)) {
        throw std::out_of_range("Flattened index out of bounds.");
    }

    return result;
}

__global__ void computeFlatIndexKernel(
    const size_t* indices, const size_t* strides,
    size_t rank, size_t* outIndex
) {
    size_t flatIndex = 0;
    for (size_t i = 0; i < rank; ++i) {
        flatIndex += indices[i] * strides[i];
    }
    *outIndex = flatIndex;
}

ValueType getValueAtIndices(
    const ValueType* deviceData,
    const size_t* hostIndices,
    const size_t* deviceStrides,
    size_t size
) {
    // Copy host indices to device
    size_t* deviceIndices;
    cudaMalloc(&deviceIndices, sizeof(size_t) * size);
    cudaMemcpy(deviceIndices, hostIndices, sizeof(size_t) * size, cudaMemcpyHostToDevice);

    // Allocate output for index
    size_t* deviceFlatIndex;
    cudaMalloc(&deviceFlatIndex, sizeof(size_t));

    // Launch kernel to compute flat index
    computeFlatIndexKernel<<<1, 1>>>(
        deviceIndices, deviceStrides, size, deviceFlatIndex
    );
    cudaDeviceSynchronize();

    // Copy back flat index
    size_t flatIndex;
    cudaMemcpy(&flatIndex, deviceFlatIndex, sizeof(size_t), cudaMemcpyDeviceToHost);

    // Get value at that index
    ValueType value;
    cudaMemcpy(&value, deviceData + flatIndex, sizeof(ValueType), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(deviceIndices);
    cudaFree(deviceFlatIndex);

    return value;
}

__global__ void setValueAtIndexKernel(ValueType* data, size_t flatIndex, ValueType value) {
    data[flatIndex] = value;
}

void setValueAtIndices(
    ValueType* deviceData,
    const size_t* hostIndices,
    const size_t* deviceStrides,
    size_t ndim,
    ValueType value
) {
    // Step 1: Allocate and copy indices to GPU
    size_t* deviceIndices;
    cudaMalloc(&deviceIndices, ndim * sizeof(size_t));
    cudaMemcpy(deviceIndices, hostIndices, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    // Step 2: Allocate memory to store computed flat index
    size_t* deviceFlatIndex;
    cudaMalloc(&deviceFlatIndex, sizeof(size_t));

    // Step 3: Launch kernel to compute flat index
    computeFlatIndexKernel<<<1, 1>>>(deviceIndices, deviceStrides, ndim, deviceFlatIndex);
    cudaDeviceSynchronize();

    // Step 4: Copy flat index to host
    size_t flatIndex;
    cudaMemcpy(&flatIndex, deviceFlatIndex, sizeof(size_t), cudaMemcpyDeviceToHost);

    // Step 5: Validate flat index
    if (flatIndex == size_t(-1)) {
        cudaFree(deviceIndices);
        cudaFree(deviceFlatIndex);
        throw std::out_of_range("Invalid indices in setValueAtIndices");
    }

    // Step 6: Launch kernel to set value at computed flat index
    setValueAtIndexKernel<<<1, 1>>>(deviceData, flatIndex, value);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(deviceIndices);
    cudaFree(deviceFlatIndex);
}

__global__ void matmulKernel(const ValueType *A, const ValueType *B, ValueType *R, size_t M, size_t K) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        ValueType sum = 0;
        for (size_t j = 0; j < K; ++j) {
            sum += A[row * K + j] * B[j];
        }
        R[row] = sum;
    }
}

__global__ void outerKernel(const ValueType *a, const ValueType *b, ValueType *result, size_t m, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = m * n;
    if (idx < total) {
        size_t i = idx / n;
        size_t j = idx % n;
        result[i * n + j] = a[i] * b[j];  // Use '=' since result is zeroed before
    }
}

__global__ void matmulTKernel(const ValueType *W, const ValueType *V, ValueType *R, size_t M, size_t N) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        ValueType sum = 0.0f;
        for (size_t i = 0; i < M; ++i) {
            sum += W[i * N + col] * V[i];
        }
        R[col] = sum;
    }
}

// Wrapper functions to launch kernels

void matmul(const ValueType *A, const ValueType *B, ValueType *R, size_t M, size_t K) {
    const int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    matmulKernel<<<gridSize, blockSize>>>(A, B, R, M, K);
    cudaDeviceSynchronize();
}

void outer(const ValueType *a, const ValueType *b, ValueType *result, size_t m, size_t n) {
    const int blockSize = 256;
    int gridSize = (m * n + blockSize - 1) / blockSize;
    outerKernel<<<gridSize, blockSize>>>(a, b, result, m, n);
    cudaDeviceSynchronize();
}

void matmulT(const ValueType *W, const ValueType *V, ValueType *R, size_t M, size_t N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    matmulTKernel<<<gridSize, blockSize>>>(W, V, R, M, N);
    cudaDeviceSynchronize();
}
} // namespace tensor_gpu
