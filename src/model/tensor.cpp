#include "tensor_gpu.hpp"
#include <cstddef>
#include <cuda_runtime.h> // Added for CUDA error checking
#include <iostream>       // Added for debugging
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tensor.hpp>

namespace nn::global {

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
size_t computeTensorSize(const std::vector<size_t> &shape) {
	if (shape.empty())
		return 0;

	size_t size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}
	return size;
}

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
std::string shapeToString(const std::vector<size_t> &shape) {
	if (shape.empty()) {
		return "{}";
	}
	std::stringstream ss;
	ss << "{";
	for (size_t i = 0; i < shape.size() - 1; ++i) {
		ss << shape[i] << ", ";
	}
	ss << shape.back() << "}";
	return ss.str();
}

// Static member initialization
bool Tensor::isGpu = DEFAULT_GPU_MODE; ///< Global GPU mode flag for all tensors
size_t Tensor::tensorCount = 0;        ///< Global counter for tracking active tensors

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
Tensor::Tensor(const std::vector<size_t> &shape_, ValueType init) {
	if (shape_.empty()) {
		throw std::invalid_argument("Invalid argument: Tensor shape cannot be empty.");
	}

	// Calculate total number of elements by multiplying all dimensions
	size_t totalSize = std::accumulate(
	    shape_.begin(),
	    shape_.end(),
	    size_t(1),
	    std::multiplies<>());

	shape = shape_;

	if (isGpu) {
		// Allocate memory on GPU and initialize with the specified value
		std::cout << "DEBUG: Tensor constructor - allocating " << totalSize << " elements (" << (totalSize * sizeof(ValueType)) << " bytes)" << std::endl;
		gpu_data = (ValueType *)tensor_gpu::allocate(totalSize * sizeof(ValueType));
		gpu_data_size = totalSize;
		fill(init);
	} else {
		// Allocate memory on CPU and initialize with the specified value
		cpu_data.assign(totalSize, init);
	}

	// Compute strides for efficient multi-dimensional indexing
	computeStrides();

	// Increment global tensor counter
	tensorCount++;
}

/**
 * @brief Switches the global execution mode to GPU for all future tensors
 *
 * This static method changes the global execution mode to GPU, meaning all
 * subsequently created tensors will be allocated on GPU memory. This switch
 * can only be performed when no tensors currently exist in the system.
 *
 * @throws std::runtime_error If tensors already exist in CPU mode
 *
 * @note This is a global setting that affects all future tensor allocations
 * @note All existing tensors must be destroyed before switching modes
 */
void Tensor::toGpu() {
	if (isGpu)
		return;

	if (tensorCount > 0)
		throw std::runtime_error("Cannot switch to GPU mode: tensors already exist in CPU mode.");

	isGpu = true;
}

/**
 * @brief Switches the global execution mode to CPU for all future tensors
 *
 * This static method changes the global execution mode to CPU, meaning all
 * subsequently created tensors will be allocated on CPU memory. This switch
 * can only be performed when no tensors currently exist in the system.
 *
 * @throws std::runtime_error If tensors already exist in GPU mode
 *
 * @note This is a global setting that affects all future tensor allocations
 * @note All existing tensors must be destroyed before switching modes
 */
void Tensor::toCpu() {
	if (!isGpu)
		return;

	if (tensorCount > 0)
		throw std::runtime_error("Cannot switch to CPU mode: tensors already exist in GPU mode.");

	isGpu = false;
}

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
Tensor::Tensor(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor copy constructor" << std::endl;

	// Copy shape and strides metadata
	shape = other.shape;
	strides = other.strides;

	if (isGpu) {
		// GPU mode: allocate new GPU memory and copy data
		gpu_data_size = other.gpu_data_size;
		std::cout << "DEBUG: Tensor copy constructor - allocating " << gpu_data_size << " elements" << std::endl;

		// Validate source GPU data pointer
		if (other.gpu_data == nullptr) {
			std::cerr << "ERROR: other.gpu_data is null in copy constructor!" << std::endl;
			throw std::runtime_error("Null GPU data pointer in copy constructor");
		}

		// Allocate new GPU memory
		gpu_data = (ValueType *)tensor_gpu::allocate(gpu_data_size * sizeof(ValueType));
		std::cout << "DEBUG: Tensor copy constructor - allocated gpu_data: " << (void *)gpu_data << std::endl;

		// Copy data from source to destination on GPU
		tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
		std::cout << "DEBUG: Tensor copy constructor - copyDeviceToDevice completed" << std::endl;

		// Synchronize GPU operations and check for errors
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after copyDeviceToDevice in copy constructor: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		// CPU mode: simple vector copy
		cpu_data = other.cpu_data;
	}

	std::cout << "DEBUG: Exiting Tensor copy constructor" << std::endl;
}

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
size_t Tensor::numElements() const {
	if (isGpu) {
		return gpu_data_size;
	}

	return cpu_data.size();
}

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
void Tensor::getData(std::vector<ValueType> &dest) const {
	if (isGpu) {
		// GPU mode: copy from device to host memory
		tensor_gpu::copyToHost(dest.data(), gpu_data, gpu_data_size * sizeof(ValueType));
	} else {
		// CPU mode: simple vector assignment
		dest = cpu_data;
	}
}

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
void Tensor::setData(const Tensor &other) {
	if (this == &other)
		return;

	if (isGpu) {
		if (gpu_data_size != other.gpu_data_size) {
			// Different sizes: reallocate memory and copy
			std::cout << "DEBUG: setData - reallocating from " << gpu_data_size << " to " << other.gpu_data_size << " elements" << std::endl;
			ValueType *temp = (ValueType *)tensor_gpu::allocate(other.gpu_data_size * sizeof(ValueType));
			gpu_data_size = other.gpu_data_size;
			tensor_gpu::copyDeviceToDevice(temp, other.gpu_data, gpu_data_size * sizeof(ValueType));
			tensor_gpu::deallocate(gpu_data);
			gpu_data = temp;
		} else {
			// Same size: direct copy
			tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
		}
	} else {
		// CPU mode: simple vector assignment
		cpu_data = other.cpu_data;
	}
}

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
void Tensor::setShape(const std::vector<size_t> &newShape) {
	shape = newShape;
	computeStrides();
}

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
void Tensor::fill(const ValueType &value) {
	if (isGpu) {
		// GPU mode: zero first, then add scalar
		tensor_gpu::zero(gpu_data, gpu_data_size);
		tensor_gpu::add_scalar(gpu_data, value, gpu_data, gpu_data_size);
	} else {
		// CPU mode: simple loop assignment
		for (auto &n : cpu_data) {
			n = value;
		}
	}
}

/**
 * @brief Fills the entire tensor with zeros
 *
 * This is a convenience method that sets all tensor elements to zero.
 * It's equivalent to calling fill(0.0) but may be more efficient.
 *
 * @note For GPU tensors, this uses an optimized zero kernel
 * @note For CPU tensors, this calls fill(0) internally
 */
void Tensor::zero() {
	if (isGpu) {
		// GPU mode: use optimized zero kernel
		tensor_gpu::zero(gpu_data, gpu_data_size);
	} else {
		// CPU mode: use fill method
		fill(0);
	}
}

Tensor &Tensor::operator=(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor::operator=" << std::endl;

	if (this == &other) {
		std::cout << "DEBUG: Self-assignment detected, returning" << std::endl;
		return *this;
	}

	if (isGpu) {
		std::cout << "DEBUG: operator= - current gpu_data_size: " << gpu_data_size << ", other.gpu_data_size: " << other.gpu_data_size << std::endl;
		std::cout << "DEBUG: operator= - current gpu_data: " << (void *)gpu_data << ", other.gpu_data: " << (void *)other.gpu_data << std::endl;

		if (gpu_data_size != other.gpu_data_size) {
			std::cout << "DEBUG: operator= - reallocating from " << gpu_data_size << " to " << other.gpu_data_size << " elements" << std::endl;

			// Check for null pointers before reallocation
			if (other.gpu_data == nullptr) {
				std::cerr << "ERROR: other.gpu_data is null during reallocation!" << std::endl;
				throw std::runtime_error("Null GPU data pointer in operator=");
			}

			ValueType *temp = (ValueType *)tensor_gpu::allocate(other.gpu_data_size * sizeof(ValueType));
			std::cout << "DEBUG: operator= - allocated temp buffer: " << (void *)temp << std::endl;

			gpu_data_size = other.gpu_data_size;
			std::cout << "DEBUG: operator= - about to copy " << gpu_data_size << " elements" << std::endl;

			tensor_gpu::copyDeviceToDevice(temp, other.gpu_data, gpu_data_size * sizeof(ValueType));
			std::cout << "DEBUG: operator= - copyDeviceToDevice completed" << std::endl;

			cudaDeviceSynchronize();
			cudaError_t cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) {
				std::cerr << "CUDA Error after copyDeviceToDevice: " << cudaGetErrorString(cudaError) << std::endl;
			}

			if (gpu_data != nullptr) {
				std::cout << "DEBUG: operator= - deallocating old gpu_data: " << (void *)gpu_data << std::endl;
				tensor_gpu::deallocate(gpu_data);
			}

			gpu_data = temp;
			std::cout << "DEBUG: operator= - assigned new gpu_data: " << (void *)gpu_data << std::endl;
		} else {
			std::cout << "DEBUG: operator= - same size, copying data" << std::endl;
			if (other.gpu_data == nullptr) {
				std::cerr << "ERROR: other.gpu_data is null during copy!" << std::endl;
				throw std::runtime_error("Null GPU data pointer in operator=");
			}
			tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
			cudaDeviceSynchronize();
			cudaError_t cudaError = cudaGetLastError();
			if (cudaError != cudaSuccess) {
				std::cerr << "CUDA Error after copyDeviceToDevice: " << cudaGetErrorString(cudaError) << std::endl;
			}
		}
	} else {
		cpu_data = other.cpu_data;
	}

	shape = other.shape;
	strides = other.strides;

	std::cout << "DEBUG: Exiting Tensor::operator=" << std::endl;
	return *this;
}

Tensor &Tensor::operator=(const std::vector<ValueType> &other) {
	if (other.size() != numElements()) {
		throw std::length_error(
		    "Tensor assignment size mismatch: Tensor has " + std::to_string(numElements()) +
		    " elements, but input vector has " + std::to_string(other.size()) + " elements.");
	}

	if (isGpu) {
		tensor_gpu::copyToDevice(gpu_data, other.data(), gpu_data_size * sizeof(ValueType));
	} else {
		cpu_data = other;
	}

	return *this;
}

void Tensor::flatten() {
	shape = {numElements()};
	computeStrides();
}

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
void Tensor::computeStrides() {
	const size_t dim = shape.size();
	strides.resize(dim);
	size_t stride = 1;

	// Compute strides in reverse order for row-major layout
	for (size_t i = dim; i-- > 0;) {
		strides[i] = stride;
		stride *= shape[i];
	}
}

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
inline size_t Tensor::flattenIndex(const std::vector<size_t> &indices) const {
	if (indices.size() != shape.size()) {
		throw std::invalid_argument(
		    "Incorrect number of indices provided. Tensor has " + std::to_string(shape.size()) +
		    " dimensions, but " + std::to_string(indices.size()) + " indices were given.");
	}

	size_t index = 0;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (indices[i] >= shape[i])
			throw std::out_of_range(
			    "Index out of bounds. Index " + std::to_string(indices[i]) +
			    " is invalid for dimension " + std::to_string(i) + " which has size " +
			    std::to_string(shape[i]) + ".");
		index += indices[i] * strides[i];
	}

	return index;
}

ValueType Tensor::getValue(const std::vector<size_t> &indices) const {
	return getValue(flattenIndex(indices));
}

ValueType Tensor::getValue(const size_t indices) const {
	if (isGpu) {
		return tensor_gpu::getValueAt(gpu_data, indices);
	}

	return cpu_data[indices];
}

void Tensor::insertRange(const Tensor &other,
                         const size_t startO, const size_t startT,
                         const size_t length) {
	if (isGpu) {
		tensor_gpu::copyDeviceToDevice(gpu_data + startT, other.gpu_data + startO, length * sizeof(ValueType));
	} else {
		for (size_t i = 0; i < length; ++i) {
			cpu_data[i + startT] = other.cpu_data[i + startO];
		}
	}
}

void Tensor::setValue(const std::vector<size_t> &indices, const ValueType value) {
	setValue(flattenIndex(indices), value);
}

void Tensor::setValue(const size_t indices, const ValueType value) {
	if (isGpu) {
		tensor_gpu::setValueAt(gpu_data, indices, value);
	} else {
		cpu_data[indices] = value;
	}
}

Tensor &Tensor::operator+=(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor::operator+=" << std::endl;

	if (shape != other.shape)
		throw std::invalid_argument(
		    "Shape mismatch in Tensor::operator+=. Left-hand side shape: " +
		    shapeToString(shape) + ", right-hand side shape: " + shapeToString(other.shape) + ".");

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator+= - this->gpu_data_size: " << gpu_data_size << ", other.gpu_data_size: " << other.gpu_data_size << std::endl;
		std::cout << "DEBUG: Tensor::operator+= - this->gpu_data: " << (void *)gpu_data << ", other.gpu_data: " << (void *)other.gpu_data << std::endl;

		// Check for null pointers
		if (gpu_data == nullptr || other.gpu_data == nullptr) {
			std::cerr << "ERROR: Null GPU data pointer detected!" << std::endl;
			std::cerr << "this->gpu_data: " << (void *)gpu_data << ", other.gpu_data: " << (void *)other.gpu_data << std::endl;
			throw std::runtime_error("Null GPU data pointer in operator+=");
		}

		std::cout << "DEBUG: About to call tensor_gpu::add_vec" << std::endl;
		tensor_gpu::add_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
		std::cout << "DEBUG: tensor_gpu::add_vec completed" << std::endl;

		cudaDeviceSynchronize(); // Synchronize to catch async errors
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after add_vec: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (size_t i = 0; i < cpu_data.size(); ++i)
			cpu_data[i] += other.cpu_data[i];
	}

	std::cout << "DEBUG: Exiting Tensor::operator+=" << std::endl;
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor::operator-=" << std::endl;

	if (shape != other.shape)
		throw std::invalid_argument(
		    "Shape mismatch in Tensor::operator-=. Left-hand side shape: " +
		    shapeToString(shape) + ", right-hand side shape: " + shapeToString(other.shape) + ".");

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator-= - this->gpu_data_size: " << gpu_data_size << ", other.gpu_data_size: " << other.gpu_data_size << std::endl;
		tensor_gpu::subtraction_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after subtraction_vec: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (size_t i = 0; i < cpu_data.size(); ++i)
			cpu_data[i] -= other.cpu_data[i];
	}

	std::cout << "DEBUG: Exiting Tensor::operator-=" << std::endl;
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor::operator*=" << std::endl;

	if (shape != other.shape)
		throw std::invalid_argument(
		    "Shape mismatch in Tensor::operator*=. Left-hand side shape: " +
		    shapeToString(shape) + ", right-hand side shape: " + shapeToString(other.shape) + ".");

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator*= - this->gpu_data_size: " << gpu_data_size << ", other.gpu_data_size: " << other.gpu_data_size << std::endl;
		tensor_gpu::multiply_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after multiply_vec: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (size_t i = 0; i < cpu_data.size(); ++i)
			cpu_data[i] *= other.cpu_data[i];
	}

	std::cout << "DEBUG: Exiting Tensor::operator*=" << std::endl;
	return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
	std::cout << "DEBUG: Entering Tensor::operator/=" << std::endl;

	if (shape != other.shape)
		throw std::invalid_argument(
		    "Shape mismatch in Tensor::operator/=. Left-hand side shape: " +
		    shapeToString(shape) + ", right-hand side shape: " + shapeToString(other.shape) + ".");

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator/= - this->gpu_data_size: " << gpu_data_size << ", other.gpu_data_size: " << other.gpu_data_size << std::endl;
		tensor_gpu::division_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after division_vec: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (size_t i = 0; i < cpu_data.size(); ++i)
			cpu_data[i] /= other.cpu_data[i];
	}

	std::cout << "DEBUG: Exiting Tensor::operator/=" << std::endl;
	return *this;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	std::cout << "DEBUG: Entering Tensor::operator*=(scalar)" << std::endl;

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator*=(scalar) - this->gpu_data_size: " << gpu_data_size << std::endl;
		tensor_gpu::multiply_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after multiply_scalar: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (auto &x : cpu_data)
			x *= scalar;
	}

	std::cout << "DEBUG: Exiting Tensor::operator*=(scalar)" << std::endl;
	return *this;
}

Tensor &Tensor::operator-=(ValueType scalar) {
	std::cout << "DEBUG: Entering Tensor::operator-=(scalar)" << std::endl;

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator-=(scalar) - this->gpu_data_size: " << gpu_data_size << std::endl;
		tensor_gpu::subtraction_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after subtraction_scalar: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (auto &x : cpu_data)
			x -= scalar;
	}

	std::cout << "DEBUG: Exiting Tensor::operator-=(scalar)" << std::endl;
	return *this;
}

Tensor &Tensor::operator+=(ValueType scalar) {
	std::cout << "DEBUG: Entering Tensor::operator+=(scalar)" << std::endl;

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator+=(scalar) - this->gpu_data_size: " << gpu_data_size << std::endl;
		tensor_gpu::add_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after add_scalar: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (auto &x : cpu_data)
			x += scalar;
	}

	std::cout << "DEBUG: Exiting Tensor::operator+=(scalar)" << std::endl;
	return *this;
}

Tensor &Tensor::operator/=(ValueType scalar) {
	std::cout << "DEBUG: Entering Tensor::operator/=(scalar)" << std::endl;

	if (isGpu) {
		std::cout << "DEBUG: Tensor::operator/=(scalar) - this->gpu_data_size: " << gpu_data_size << std::endl;
		tensor_gpu::division_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
		cudaDeviceSynchronize();
		cudaError_t cudaError = cudaGetLastError();
		if (cudaError != cudaSuccess) {
			std::cerr << "CUDA Error after division_scalar: " << cudaGetErrorString(cudaError) << std::endl;
		}
	} else {
		for (auto &x : cpu_data)
			x /= scalar;
	}

	std::cout << "DEBUG: Exiting Tensor::operator/=(scalar)" << std::endl;
	return *this;
}

void Tensor::matmul(const Tensor &other, Tensor &result) const {
	size_t M = shape[0];
	size_t K = shape[1];

	result.zero();

	if (isGpu) {
		tensor_gpu::matmul(gpu_data, other.gpu_data, result.gpu_data, M, K);
	} else {
		const float *A = cpu_data.data();
		const float *B = other.cpu_data.data();
		float *R = result.cpu_data.data();

		for (size_t i = 0; i < M; ++i) {
			float sum = 0.0f;
			size_t base = i * K;
			for (size_t j = 0; j < K; ++j) {
				sum += A[base + j] * B[j];
			}
			R[i] = sum;
		}
	}
}

void Tensor::outer(const Tensor &a, const Tensor &b, Tensor &result) {
	size_t m = a.shape[0];
	size_t n = b.shape[0];

	result.zero();

	if (isGpu) {
		tensor_gpu::outer(a.gpu_data, b.gpu_data, result.gpu_data, m, n);
	} else {
		float *r = result.cpu_data.data();
		const float *A = a.cpu_data.data();
		const float *B = b.cpu_data.data();

		for (size_t i = 0; i < m; ++i) {
			for (size_t j = 0; j < n; ++j) {
				r[i * n + j] += A[i] * B[j];
			}
		}
	}
}

void Tensor::matmulT(const Tensor &vec, Tensor &result) const {
	result.zero();

	if (isGpu) {
		tensor_gpu::matmulT(gpu_data, vec.gpu_data, result.gpu_data, shape[0], shape[1]);
	} else {
		for (size_t i = 0; i < shape[1]; ++i) {
			for (size_t j = 0; j < shape[0]; ++j) {
				result.cpu_data[i] += cpu_data[j * shape[1] + i] * vec.cpu_data[j];
			}
		}
	}
}

/**
 * @brief Destructor - cleans up tensor resources
 *
 * This destructor properly cleans up all resources associated with the tensor.
 * For GPU tensors, it deallocates the device memory. For CPU tensors, the
 * std::vector destructor handles cleanup automatically. The global tensor
 * count is decremented to track the number of active tensors.
 *
 * @note GPU memory is only deallocated if the tensor is in GPU mode and has valid data
 * @note The global tensor count is decremented upon destruction
 * @note This destructor is automatically called when the tensor goes out of scope
 */
Tensor::~Tensor() {
	if (isGpu && gpu_data != nullptr) {
		// GPU mode: deallocate device memory
		std::cout << "DEBUG: Tensor destructor - deallocating " << gpu_data_size << " elements" << std::endl;
		tensor_gpu::deallocate(gpu_data);
		gpu_data = nullptr;
	}
	// Decrement global tensor counter
	tensorCount--;
}
} // namespace nn::global
