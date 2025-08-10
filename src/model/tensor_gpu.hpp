#ifndef TENSOR_GPU
#define TENSOR_GPU

#include <cstddef>

namespace nn::global {
using ValueType = float;
}

class Tensor; // Forward declaration

namespace nn::global::tensor_gpu {

// ============================
// Memory Management
// ============================
void *allocate(std::size_t size);
void deallocate(void *devicePtr);

void copyToDevice(void *deviceDst, const void *hostSrc, std::size_t count);
void copyToHost(void *hostDst, const void *deviceSrc, std::size_t count);
void copyDeviceToDevice(void *deviceDst, const void *deviceSrc, std::size_t count);

void zero(ValueType *deviceData, std::size_t count);

// ============================
// Element-wise Operations (Vector-Vector)
// ============================
void add_vec(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);
void subtraction_vec(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);
void division_vec(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);
void multiply_vec(const ValueType *A, const ValueType *B, ValueType *C, std::size_t count);

// ============================
// Element-wise Operations (Vector-Scalar)
// ============================
void add_scalar(const ValueType *A, ValueType B, ValueType *C, std::size_t count);
void subtraction_scalar(const ValueType *A, ValueType B, ValueType *C, std::size_t count);
void division_scalar(const ValueType *A, ValueType B, ValueType *C, std::size_t count);
void multiply_scalar(const ValueType *A, ValueType B, ValueType *C, std::size_t count);

// ============================
// Activation Functions
// ============================

// ReLU
void relu(const ValueType *input, ValueType *output, std::size_t count);
void relu_derivative(const ValueType *input, ValueType *output, std::size_t count);

// Leaky ReLU
void leaky_relu(const ValueType *input, ValueType *output, std::size_t count, ValueType alpha = 0.01f);
void leaky_relu_derivative(const ValueType *input, ValueType *output, std::size_t count, ValueType alpha = 0.01f);

// Sigmoid
void sigmoid(const ValueType *input, ValueType *output, std::size_t count);
void sigmoid_derivative(const ValueType *input, ValueType *output, std::size_t count);

// Tanh
void tanh_activation(const ValueType *input, ValueType *output, std::size_t count);
void tanh_derivative(const ValueType *input, ValueType *output, std::size_t count);

// Softmax
void softmax(const ValueType *net, ValueType *out, std::size_t size);

// ============================
// Single Value Access
// ============================
ValueType getValueAt(const ValueType *devicePtr, std::size_t index);
void setValueAt(ValueType *devicePtr, std::size_t index, ValueType value);

// ============================
// Matrix Operations
// ============================
void matmul(const ValueType *A, const ValueType *B, ValueType *R, std::size_t M, std::size_t K);
void matmulT(const ValueType *W, const ValueType *V, ValueType *R, std::size_t M, std::size_t N);
void outer(const ValueType *a, const ValueType *b, ValueType *result, std::size_t m, std::size_t n);

struct MaxIndex {
	ValueType value;
	std::size_t index;
};
std::size_t getMaxElementIndex(const ValueType* deviceData, std::size_t count);
} // namespace nn::global::tensor_gpu

#endif // TENSOR_GPU
