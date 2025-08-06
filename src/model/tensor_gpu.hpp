#ifndef TENSOR_GPU
#define TENSOR_GPU

#include <cstddef>
namespace nn::global {
using ValueType = float;
}

class Tensor; // Forward declaration

namespace nn::global::tensor_gpu {

/// Allocate memory on GPU for a tensor.
void *allocate(std::size_t count);

/// Free GPU memory.
void deallocate(ValueType *devicePtr);

/// Copy data from CPU to GPU.
void copyToDevice(void *deviceDst, const void *hostSrc, std::size_t count);

/// Copy data from GPU to CPU.
void copyToHost(void *hostDst, const void *deviceSrc, std::size_t count);

void copyDeviceToDevice(void *deviceDst, const void *deviceSrc, std::size_t count);

/// Set all elements to zero (on GPU).
void zero(ValueType *deviceData, std::size_t count);

/// Element-wise addition: C = A + B
void add(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise multiply: C = A * B
void multiply(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Dot product between two vectors (A · B)
float dot(const ValueType *A, const ValueType *B, std::size_t count);

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

void softmax(const ValueType *net, ValueType *out, std::size_t size);

template <typename T>
ValueType getValueAt(const T *devicePtr, std::size_t index);

template <typename T>
void setValueAt(T *devicePtr, std::size_t index, T value);

size_t flattenIndexGpu(const size_t *indices, const size_t *d_shape, const size_t *d_strides, size_t ndim);

} // namespace nn::global::tensor_gpu

#endif // TENSOR_GPU
