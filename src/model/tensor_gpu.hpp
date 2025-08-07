#ifndef TENSOR_GPU
#define TENSOR_GPU

#include <cstddef>

namespace nn::global {
using ValueType = float;
}

class Tensor; // Forward declaration

namespace nn::global::tensor_gpu {

/// Allocate memory on GPU for a tensor.
void *allocate(std::size_t size);

/// Free GPU memory.
void deallocate(void *devicePtr);

/// Copy data from CPU to GPU.
void copyToDevice(void *deviceDst, const void *hostSrc, std::size_t count);

/// Copy data from GPU to CPU.
void copyToHost(void *hostDst, const void *deviceSrc, std::size_t count);

void copyDeviceToDevice(void *deviceDst, const void *deviceSrc, std::size_t count);

/// Set all elements to zero (on GPU).
void zero(ValueType *deviceData, std::size_t count);

/// Element-wise addition: C = A + B
void add(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise addition: C = A - B
void subtraction(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise addition: C = A / B
void division(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise multiply: C = A * B
void multiply(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise addition: C = A + B
void add(const ValueType *A, const ValueType B, ValueType *C, std::size_t count);

/// Element-wise addition: C = A - B
void subtraction(const ValueType *A, const ValueType B, ValueType *C, std::size_t count);

/// Element-wise addition: C = A / B
void division(const ValueType *A, const ValueType B, ValueType *C, std::size_t count);

/// Element-wise multiply: C = A * B
void multiply(const ValueType *A, const ValueType B, ValueType *C, std::size_t count);

void computeStridesDevice(const size_t *gpu_shape, size_t *gpu_strides, std::size_t ndim);

// ---------------- ReLU ----------------
void relu(const ValueType *input, ValueType *output, std::size_t count);
void relu_derivative(const ValueType *input, ValueType *output, std::size_t count);

// ---------------- Sigmoid ----------------
void sigmoid(const ValueType *input, ValueType *output, std::size_t count);
void sigmoid_derivative(const ValueType *input, ValueType *output, std::size_t count);

// ---------------- Tanh ----------------
void tanh_activation(const ValueType *input, ValueType *output, std::size_t count);
void tanh_derivative(const ValueType *input, ValueType *output, std::size_t count);

// ---------------- Leaky ReLU ----------------
void leaky_relu(const ValueType *input, ValueType *output, std::size_t count, ValueType alpha = 0.01f);
void leaky_relu_derivative(const ValueType *input, ValueType *output, std::size_t count, ValueType alpha = 0.01f);

// ---------------- Softmax ----------------
void softmax(const ValueType *net, ValueType *out, std::size_t size);

ValueType getValueAt(const ValueType *devicePtr, std::size_t index);

void setValueAt(ValueType *devicePtr, std::size_t index, ValueType value);

size_t flattenIndexGpu(const size_t *indices, const size_t *d_shape, const size_t *d_strides, size_t ndim);

ValueType getValueAtIndices(
    const ValueType *deviceData,
    const size_t *hostIndices,
    const size_t *deviceStrides,
    size_t size);

void setValueAtIndices(
    ValueType *deviceData,
    const size_t *hostIndices,
    const size_t *deviceStrides,
    size_t ndim,
    ValueType value);

void matmul(const ValueType *A, const ValueType *B, ValueType *R, size_t M, size_t K);
void outer(const ValueType *a, const ValueType *b, ValueType *result, size_t m, size_t n);
void matmulT(const ValueType *W, const ValueType *V, ValueType *R, size_t M, size_t N);
} // namespace nn::global::tensor_gpu

#endif // TENSOR_GPU
