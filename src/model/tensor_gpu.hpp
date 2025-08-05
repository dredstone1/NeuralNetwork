#include <cstddef>

class Tensor; // Forward declaration

namespace tensor_gpu {

/// Allocate memory on GPU for a tensor.
float *allocate(std::size_t count);

/// Free GPU memory.
void deallocate(float *devicePtr);

/// Copy data from CPU to GPU.
void copyToDevice(float *deviceDst, const float *hostSrc, std::size_t count);

/// Copy data from GPU to CPU.
void copyToHost(float *hostDst, const float *deviceSrc, std::size_t count);

/// Set all elements to zero (on GPU).
void zero(float *deviceData, std::size_t count);

/// Element-wise addition: C = A + B
void add(const float *A, const float *B, float *C, std::size_t count);

/// Element-wise multiply: C = A * B
void multiply(const float *A, const float *B, float *C, std::size_t count);

/// Dot product between two vectors (A · B)
float dot(const float *A, const float *B, std::size_t count);

/// Apply activation function (e.g., ReLU)
void relu(float *deviceData, std::size_t count);

/// Apply derivative of activation function (e.g., ReLU')
void relu_derivative(const float *input, float *output, std::size_t count);

// Add more operations as needed...

} // namespace tensor_gpu
