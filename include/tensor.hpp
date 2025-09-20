#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "../src/model/tensor_gpu.hpp"
#include <string>
#include <vector>

// Forward declarations
namespace nn::model {
class Activation;
class DataBase;
class IOptimizer;
namespace cnn {
class CNNetwork;
}
} // namespace nn::model

namespace nn::global {

class Tensor;

/**
 * @brief Converts a tensor shape into a human-readable string representation
 *
 * This utility function formats a shape vector into a string that can be used
 * for debugging, error messages, or logging purposes. The output format is
 * similar to array notation: "{dim1, dim2, dim3, ...}".
 *
 * @param shape A vector containing the dimensions of the tensor
 * @return A formatted string representation of the shape
 * @retval "{}" If the shape vector is empty
 */

std::string shapeToString(const std::vector<size_t> &shape);

/**
 * @brief Computes the total number of elements in a tensor from its shape
 *
 * This function calculates the product of all dimensions in the shape vector,
 * which represents the total number of elements that can be stored in a tensor
 * with the given dimensions.
 *
 * @param shape A vector containing the dimensions of the tensor
 * @return The total number of elements (product of all dimensions)
 * @retval 0 If the shape vector is empty
 */
size_t computeTensorSize(const std::vector<size_t> &shape);

/// Default GPU state for the application.
constexpr bool DEFAULT_GPU_MODE = false;

/// Default initialization value for tensor elements.
constexpr ValueType DEFAULT_INIT_VALUE = 0.0f;

/**
 * @class Tensor
 * @brief Multi-dimensional array supporting CPU and GPU backends.
 *
 * `Tensor` is the core data structure for neural network computations.
 * It provides n-dimensional storage and backend-agnostic execution.
 *
 * @section features Core Features
 * - N-dimensional data storage with row-major layout
 * - Element-wise arithmetic with tensors and scalars
 * - Linear algebra operations (matrix-vector multiplication, outer products)
 * - Seamless backend switching between CPU and GPU
 *
 * @section execution_model Execution Model (CPU/GPU)
 * Global execution mode is shared across all `Tensor` instances.
 * Switching between CPU and GPU is only allowed when no tensors exist.
 * Steps:
 * 1. Switch mode using `Tensor::toGpu()` or `Tensor::toCpu()`.
 * 2. Create and use `Tensor` objects.
 * 3. **Destroy all `Tensor` objects** before switching mode again.
 *
 * @warning Switching modes with existing tensors throws a `std::runtime_error`.
 *
 * @note Certain model classes (e.g., Activation, DataBase, CNNetwork) have direct access to tensor data for efficiency.
 */
class Tensor {
  private:
	std::vector<ValueType> cpu_data;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	ValueType *gpu_data = nullptr;
	std::size_t gpu_data_size = 0;

	static bool isGpu;
	static size_t tensorCount;

	/**
	 * @brief Computes the strides for efficient multi-dimensional indexing
	 *
	 * This method calculates the stride values for each dimension, which are used
	 * to convert multi-dimensional indices into a single flattened index. The
	 * strides are computed using row-major (C-style) ordering.
	 *
	 * @note This is a private method called internally when shape changes
	 * @note Strides are computed as: stride[i] = product of all dimensions after i
	 */
	void computeStrides();

	/**
	 * @brief Converts multi-dimensional indices to a single flattened index
	 *
	 * This method takes a vector of indices (one for each dimension) and converts
	 * them into a single linear index that can be used to access the underlying
	 * data array. The conversion uses row-major (C-style) ordering.
	 *
	 * @param indices A vector of indices, one for each dimension
	 * @return The corresponding flattened index
	 *
	 * @throws std::invalid_argument If the number of indices doesn't match the tensor's rank
	 * @throws std::out_of_range If any index is out of bounds for its dimension
	 *
	 * @note This is a private method used internally for element access
	 * @note The conversion formula is: index = sum(indices[i] * strides[i])
	 */
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

	// Friend classes for direct data access
	friend model::Activation;
	friend model::DataBase;
	friend nn::model::cnn::CNNetwork;
    friend nn::model::IOptimizer;

  public:
	// --- Constructors, Destructor, and Assignment ---

	/**
	 * @brief Constructs a new tensor with the specified shape and initial value
	 *
	 * This constructor creates a tensor with the given dimensions and initializes
	 * all elements with the specified value. The tensor will be allocated either
	 * on CPU or GPU memory depending on the current global execution mode.
	 *
	 * @param shape_ The dimensions of the tensor (e.g., {batch_size, height, width, channels})
	 * @param init The initial value to fill all tensor elements with (default: 0.0)
	 *
	 * @throws std::invalid_argument If the shape vector is empty
	 *
	 * @note The tensor count is incremented upon successful construction
	 * @note GPU memory allocation is performed if the global GPU mode is enabled
	 */
	Tensor(const std::vector<size_t> &shape, ValueType init = DEFAULT_INIT_VALUE);

	/**
	 * @brief Copy constructor - creates a deep copy of another tensor
	 *
	 * This constructor creates a new tensor that is an exact copy of the source tensor,
	 * including all data, shape, and strides. The copy is performed in the same execution
	 * mode (CPU/GPU) as the source tensor.
	 *
	 * @param other The tensor to copy from
	 *
	 * @throws std::runtime_error If GPU data pointer is null during GPU copy operation
	 *
	 * @note This performs a deep copy - the new tensor has its own memory allocation
	 * @note GPU operations are synchronized to ensure completion before proceeding
	 * @note The tensor count is incremented upon successful construction
	 */
	Tensor(const Tensor &other);

	/**
	 * @brief Destructor. Frees GPU memory if allocated and decrements the global tensor count.
	 */
	~Tensor();

	/**
	 * @brief Assignment operator. Replaces this tensor's data and shape with another's.
	 * @param other The tensor to assign from.
	 * @return A reference to this tensor.
	 */
	Tensor &operator=(const Tensor &other);

	/**
	 * @brief Assigns data from a `std::vector` to this tensor.
	 * @param other The vector containing the new data.
	 * @return A reference to this tensor.
	 * @throws std::length_error If the vector size does not match the tensor's total elements.
	 */
	Tensor &operator=(const std::vector<ValueType> &other);

	// --- Element Access ---

	/**
	 * @brief Retrieves the value at the specified multi-dimensional indices.
	 * @param indices A vector of indices, one for each dimension.
	 * @return The value at the given position.
	 * @throws std::invalid_argument If the number of indices does not match the tensor's rank.
	 * @throws std::out_of_range If any index is out of bounds.
	 */
	ValueType getValue(const std::vector<size_t> &indices) const;

	/**
	 * @brief Retrieves the value at a specified flattened index.
	 * @param index The 1D index into the tensor's data array.
	 * @return The value at the given position.
	 */
	ValueType getValue(const size_t index) const;

	/**
	 * @brief Sets the value at the specified multi-dimensional indices.
	 * @param indices A vector of indices, one for each dimension.
	 * @param value The new value to set.
	 */
	void setValue(const std::vector<size_t> &indices, const ValueType value);

	/**
	 * @brief Sets the value at a specified flattened index.
	 * @param index The 1D index into the tensor's data array.
	 * @param value The new value to set.
	 */
	void setValue(const size_t index, const ValueType value);

	// --- Data Manipulation ---

	/**
	 * @brief Copies a range of elements from another tensor into this one.
	 * @param other The source tensor.
	 * @param startO The starting index in the source tensor (`other`).
	 * @param startT The starting index in this tensor (the target).
	 * @param length The number of elements to copy.
	 */
	void insertRange(const Tensor &other, const size_t startO,
	                 const size_t startT, const size_t length);

	/**
	 * @brief Copies the tensor's data into a std::vector
	 *
	 * This method extracts all data from the tensor and copies it into the provided
	 * vector. For GPU tensors, this involves a device-to-host memory transfer.
	 *
	 * @param dest The destination vector that will receive the tensor data
	 *
	 * @note The destination vector will be resized to fit the tensor data
	 * @note For GPU tensors, this operation involves CUDA memory transfer
	 * @note The destination vector must have sufficient capacity or will be resized
	 */
	void getData(std::vector<ValueType> &dest) const;

	/**
	 * @brief Replaces the tensor's data with data from another tensor
	 *
	 * This method copies all data from the source tensor into this tensor. If the
	 * tensors have different sizes, memory will be reallocated as needed.
	 *
	 * @param other The source tensor to copy data from
	 *
	 * @note This method performs a no-op if called with the same tensor (self-assignment)
	 * @note For GPU tensors, this involves device-to-device memory copy
	 * @note Memory reallocation occurs if the source tensor has different size
	 */
	void setData(const Tensor &other);

	/**
	 * @brief Fills the entire tensor with a specified scalar value.
	 * @param value The value to fill the tensor with.
	 */
	void fill(const ValueType &value);

	/**
	 * @brief Fills the entire tensor with a specified scalar value
	 *
	 * This method sets all elements in the tensor to the same value. The operation
	 * is optimized for both CPU and GPU execution modes.
	 *
	 * @param value The value to fill all tensor elements with
	 *
	 * @note For GPU tensors, this uses optimized CUDA kernels
	 * @note For CPU tensors, this uses a simple loop
	 */
	void zero();

	// --- Shape and Dimensionality ---

	/**
	 * @brief Returns the total number of elements in the tensor
	 *
	 * This method returns the total number of elements that can be stored in the tensor,
	 * which is the product of all dimensions in the shape vector.
	 *
	 * @return The total number of elements in the tensor
	 *
	 * @note For GPU tensors, this returns the stored gpu_data_size
	 * @note For CPU tensors, this returns the size of the cpu_data vector
	 */
	size_t numElements() const;

	/**
	 * @brief Gets the shape of the tensor.
	 * @return A const reference to the shape vector.
	 */
	const std::vector<size_t> &getShape() const { return shape; }

	/**
	 * @brief Gets the strides of the tensor.
	 * @return A const reference to the strides vector.
	 */
	const std::vector<size_t> &getStrides() const { return strides; }

	/**
	 * @brief Reshapes the tensor into a 1D vector (flattens it).
	 * @note This only modifies the shape metadata; the underlying data is unchanged.
	 */
	void flatten();

	/**
	 * @brief Sets a new shape for the tensor
	 *
	 * This method changes the shape of the tensor and recomputes the strides
	 * for efficient multi-dimensional indexing.
	 *
	 * @param newShape The new shape dimensions for the tensor
	 *
	 * @warning The total number of elements defined by newShape must exactly match
	 *          the current number of elements. This function does not validate this.
	 *
	 * @note This only modifies the shape metadata; the underlying data is unchanged
	 * @note Strides are automatically recomputed after shape change
	 */
	void setShape(const std::vector<size_t> &newShape);

	// --- Raw Data Access (Advanced) ---

	/**
	 * @brief Gets a raw pointer to the GPU data buffer.
	 * @warning For advanced use only. The caller must not deallocate this pointer.
	 *          Returns `nullptr` if the tensor is in CPU mode.
	 * @return A raw pointer to the device memory.
	 */
	ValueType *getGpuDataP() const { return gpu_data; }

	/**
	 * @brief Gets a reference to the underlying CPU data vector.
	 * @warning For advanced use only. Throws if the tensor is in GPU mode.
	 * @return A mutable reference to the `std::vector` holding the data.
	 */
	std::vector<ValueType> &getCpuDataP() { return cpu_data; }

	// --- Arithmetic Operators (In-place) ---

	/// @{
	/**
	 * @brief Performs element-wise addition and assigns the result to this tensor.
	 * @param other The tensor to add. Must have the same shape.
	 * @return A reference to this modified tensor.
	 */
	Tensor &operator+=(const Tensor &other);
	Tensor &operator-=(const Tensor &other);
	Tensor &operator*=(const Tensor &other);
	Tensor &operator/=(const Tensor &other);
	/// @}

	/// @{
	/**
	 * @brief Performs element-wise operation with a scalar and assigns the result.
	 * @param scalar The scalar value to use in the operation.
	 * @return A reference to this modified tensor.
	 */
	Tensor &operator+=(ValueType scalar);
	Tensor &operator-=(ValueType scalar);
	Tensor &operator*=(ValueType scalar);
	Tensor &operator/=(ValueType scalar);
	/// @}

	// --- Linear Algebra ---

	/**
	 * @brief Performs matrix-vector multiplication.
	 *
	 * Computes `result = this * other`, where `this` is a matrix and `other` is a vector.
	 * - `this` shape: `{M, K}`
	 * - `other` shape: `{K}`
	 * - `result` shape: `{M}`
	 *
	 * @param other The vector to multiply with.
	 * @param result The tensor to store the output vector. It will be zeroed before the operation.
	 */
	void matmul(const Tensor &other, Tensor &result) const;

	/**
	 * @brief Computes the outer product of two vectors.
	 *
	 * Computes `result = a * b^T`, where `a` and `b` are vectors.
	 * - `a` shape: `{M}`
	 * - `b` shape: `{N}`
	 * - `result` shape: `{M, N}`
	 *
	 * @param a The first vector.
	 * @param b The second vector.
	 * @param result The tensor to store the output matrix. It will be zeroed before the operation.
	 */
	static void outer(const Tensor &a, const Tensor &b, Tensor &result);

	/**
	 * @brief Performs matrix-vector multiplication with the transpose of this tensor.
	 *
	 * Computes `result = this^T * vec`.
	 * - `this` (matrix) shape: `{M, K}`
	 * - `vec` shape: `{M}`
	 * - `result` shape: `{K}`
	 *
	 * @param vec The vector to multiply with.
	 * @param result The tensor to store the output vector. It will be zeroed before the operation.
	 */
	void matmulT(const Tensor &vec, Tensor &result) const;

	// --- Global CPU/GPU Control ---

	/**
	 * @brief Switches the global execution mode to GPU for all future tensors.
	 * @warning This method can only be called when no `Tensor` instances exist.
	 * @throws std::runtime_error If tensors already exist or if CUDA is not supported.
	 */
	static void toGpu();

	/**
	 * @brief Switches the global execution mode to CPU for all future tensors.
	 * @warning This method can only be called when no `Tensor` instances exist.
	 * @throws std::runtime_error If tensors already exist.
	 */
	static void toCpu();

	/**
	 * @brief Checks if the current global execution mode is GPU.
	 * @return `true` if in GPU mode, `false` otherwise.
	 */
	static bool getGpuState() { return isGpu; }
};

} // namespace nn::global

#endif // TENSOR_HPP
