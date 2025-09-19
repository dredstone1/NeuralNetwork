#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "../src/model/tensor_gpu.hpp"
#include <string>
#include <vector>

// Forward declarations
namespace nn::model {
class Activation;
class DataBase;
namespace cnn {
class CNNetwork;
}
} // namespace nn::model

namespace nn::global {

class Tensor;

/**
 * @brief Converts a tensor shape into a human-readable string.
 * @param shape A vector representing the dimensions of the tensor.
 * @return A string representation, e.g., "{3, 4, 5}".
 */
std::string shapeToString(const std::vector<size_t> &shape);

/**
 * @brief Computes the total number of elements in a tensor from its shape.
 * @param shape The dimensions of the tensor.
 * @return The product of all dimensions in the shape.
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

	/// Computes the strides based on the current shape for row-major layout.
	void computeStrides();

	/// Converts multi-dimensional indices to a single flattened index.
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

	// Friend classes for direct data access
	friend model::Activation;
	friend model::DataBase;
	friend nn::model::cnn::CNNetwork;

  public:
	// --- Constructors, Destructor, and Assignment ---

	/**
	 * @brief Constructs a new tensor with a given shape and initial value.
	 * @param shape The dimensions of the tensor (e.g., {batches, channels, height, width}).
	 * @param init The value to initialize all elements with.
	 * @throws std::invalid_argument If the shape is empty.
	 */
	Tensor(const std::vector<size_t> &shape, ValueType init = DEFAULT_INIT_VALUE);

	/**
	 * @brief Copy constructor. Creates a deep copy of another tensor.
	 * @param other The tensor to copy.
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
	 * @brief Copies the tensor's data into a `std::vector`.
	 * @note If in GPU mode, this involves a device-to-host memory transfer.
	 * @param dest The destination vector. It will be resized to fit the data.
	 */
	void getData(std::vector<ValueType> &dest) const;

	/**
	 * @brief Replaces the tensor's data with data from another tensor.
	 * @param other The source tensor.
	 * @throws std::length_error If the source and destination tensors have different numbers of elements.
	 */
	void setData(const Tensor &other);

	/**
	 * @brief Fills the entire tensor with a specified scalar value.
	 * @param value The value to fill the tensor with.
	 */
	void fill(const ValueType &value);

	/**
	 * @brief Fills the entire tensor with zeros. A convenience wrapper for `fill(0.0)`.
	 */
	void zero();

	// --- Shape and Dimensionality ---

	/**
	 * @brief Returns the total number of elements in the tensor.
	 * @return The product of all dimensions.
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
	 * @brief Sets a new shape for the tensor.
	 * @warning The total number of elements defined by `newShape` must exactly match
	 *          the current number of elements. This function does not validate this.
	 * @param newShape The new shape for the tensor.
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
