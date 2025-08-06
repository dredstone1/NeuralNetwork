#include <cuda_runtime.h>
#include "tensor_gpu.hpp"
#include <cstddef>
#include <stdexcept>

namespace nn::global::tensor_gpu {
// Allocate memory on GPU for a tensor.
void* allocate(std::size_t count) {
    void* devicePtr = nullptr;
    cudaError_t err = cudaMalloc(&devicePtr, count);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
    return devicePtr;
}

// Free GPU memory.
void deallocate(ValueType* devicePtr) {
    if (devicePtr) {
        cudaFree(devicePtr);
    }
}

// Copy data from CPU to GPU.
void copyToDevice(void* deviceDst, const void * hostSrc, std::size_t size) {
    cudaMemcpy(deviceDst, hostSrc, size, cudaMemcpyHostToDevice);
}


void copyDeviceToDevice(void *deviceDst, const void *deviceSrc, std::size_t count) {
    cudaMemcpy(deviceDst, deviceSrc, count, cudaMemcpyDeviceToDevice);
}

// Copy data from GPU to CPU.
void copyToHost(void* hostDst, const void* deviceSrc, std::size_t count) {
    cudaMemcpy(hostDst, deviceSrc, count, cudaMemcpyDeviceToHost);
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
    cudaDeviceSynchronize();
}

// Kernel for element-wise addition: C = A + B
__global__ void addKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}

// Element-wise addition: C = A + B
void add(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    cudaDeviceSynchronize();
}

// Kernel for element-wise multiplication: C = A * B
__global__ void multiplyKernel(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] * B[idx];
    }
}

// Element-wise multiply: C = A * B
void multiply(const ValueType* A, const ValueType* B, ValueType* C, std::size_t count) {
    std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(A, B, C, count);
    cudaDeviceSynchronize();
}

// Dot product kernel using parallel reduction (simplified version)
__global__ void dotKernel(const ValueType* A, const ValueType* B, ValueType* partialSum, std::size_t count) {
    __shared__ ValueType cache[256];
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
float dot(const ValueType* A, const ValueType* B, std::size_t count) {
    const std::size_t blockSize = 256;
    std::size_t numBlocks = (count + blockSize - 1) / blockSize;

    // Allocate partial sums
    ValueType* d_partialSum = nullptr;
    cudaMalloc(&d_partialSum, numBlocks * sizeof(ValueType));

    dotKernel<<<numBlocks, blockSize>>>(A, B, d_partialSum, count);
    cudaDeviceSynchronize();

    // Copy partial sums to host
    ValueType* h_partialSum = new ValueType[numBlocks];
    cudaMemcpy(h_partialSum, d_partialSum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    ValueType totalSum = 0.0f;
    for (std::size_t i = 0; i < numBlocks; i++) {
        totalSum += h_partialSum[i];
    }

    delete[] h_partialSum;
    cudaFree(d_partialSum);
    return totalSum;
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
    cudaDeviceSynchronize(); // Ensure computation completes
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}

__global__ void softmaxKernel(const ValueType* input, ValueType* output, std::size_t count) {
    extern __shared__ ValueType shared[];

    std::size_t tid = threadIdx.x;
    std::size_t idx = blockIdx.x * blockDim.x + tid;

    if (idx >= count) return;

    // Load input into shared memory
    shared[tid] = input[idx];
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
    cudaDeviceSynchronize();
}

template <typename T>
void setValueAt(T* devicePtr, std::size_t index, T value) {
    cudaMemcpy(devicePtr + index, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
ValueType getValueAt(const T* devicePtr , std::size_t index) {
    T value;
    cudaMemcpy(&value, devicePtr + index, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

// Compute flattened index on device
__global__ void flattenIndexKernel(const size_t* indices, const size_t* shape, const size_t* strides, size_t ndim, size_t* outIndex) {
    size_t idx = 0;
    for (size_t i = 0; i < ndim; ++i) {
        if (indices[i] >= shape[i]) {
            *outIndex = size_t(-1); // invalid index
            return;
        }
        idx += indices[i] * strides[i];
    }
    *outIndex = idx;
}

size_t flattenIndexGpu(const size_t* indices,const size_t* d_shape,const size_t* d_strides,size_t ndim) {
    // Copy indices vector to device memory
    size_t* d_indices = nullptr;
    cudaMalloc(&d_indices, ndim * sizeof(size_t));
    cudaMemcpy(d_indices, indices, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    size_t* d_outIndex = nullptr;
    cudaMalloc(&d_outIndex, sizeof(size_t));

    // Launch kernel with a single thread since this is a scalar computation
    flattenIndexKernel<<<1, 1>>>(d_indices, d_shape, d_strides, ndim, d_outIndex);
    cudaDeviceSynchronize();

    size_t hostIndex;
    cudaMemcpy(&hostIndex, d_outIndex, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_outIndex);

    if (hostIndex == size_t(-1)) {
        throw std::out_of_range("Index out of bounds.");
    }

    return hostIndex;
}

__global__ void computeFlatIndexKernel(
    const size_t* indices, const size_t* shape, const size_t* strides,
    size_t rank, size_t* outIndex
) {
    size_t flatIndex = 0;
    for (size_t i = 0; i < rank; ++i) {
        flatIndex += indices[i] * strides[i];
    }
    *outIndex = flatIndex;
}

ValueType getValueAtIndicesGpu(
    const ValueType* deviceData,
    const size_t* hostIndices,
    const size_t* deviceShape,
    const size_t* deviceStrides,
    size_t rank
) {
    // Copy host indices to device
    size_t* deviceIndices;
    cudaMalloc(&deviceIndices, sizeof(size_t) * rank);
    cudaMemcpy(deviceIndices, hostIndices, sizeof(size_t) * rank, cudaMemcpyHostToDevice);

    // Allocate output for index
    size_t* deviceFlatIndex;
    cudaMalloc(&deviceFlatIndex, sizeof(size_t));

    // Launch kernel to compute flat index
    computeFlatIndexKernel<<<1, 1>>>(
        deviceIndices, deviceShape, deviceStrides, rank, deviceFlatIndex
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
} // namespace tensor_gpu
