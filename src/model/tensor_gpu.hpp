#ifndef TENSOR_GPU
#define TENSOR_GPU

#include "tensor.hpp"
#include <cstddef>

class Tensor; // Forward declaration

namespace nn::global::tensor_gpu {

/// Allocate memory on GPU for a tensor.
ValueType *allocate(std::size_t count);

/// Free GPU memory.
void deallocate(ValueType *devicePtr);

/// Copy data from CPU to GPU.
void copyToDevice(ValueType *deviceDst, const ValueType *hostSrc, std::size_t count);

/// Copy data from GPU to CPU.
void copyToHost(ValueType *hostDst, const ValueType *deviceSrc, std::size_t count);

/// Set all elements to zero (on GPU).
void zero(ValueType *deviceData, std::size_t count);

/// Element-wise addition: C = A + B
void add(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Element-wise multiply: C = A * B
void multiply(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

/// Dot product between two vectors (A · B)
float dot(const ValueType *A, const ValueType *B, std::size_t count);

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

} // namespace nn::global::tensor_gpu

#endif // TENSOR_GPU
