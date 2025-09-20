#include "tensor_gpu.hpp"
#include <cstddef>
#include <cuda_runtime.h> // Added for CUDA error checking
#include <iostream>       // Added for debugging
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tensor.hpp>

namespace nn::global {

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

size_t computeTensorSize(const std::vector<size_t> &shape) {
	if (shape.empty())
		return 0;

	size_t size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}
	return size;
}

// Static member initialization
bool Tensor::isGpu = DEFAULT_GPU_MODE; ///< Global GPU mode flag for all tensors
size_t Tensor::tensorCount = 0;        ///< Global counter for tracking active tensors

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

ValueType Tensor::getValue(const std::vector<size_t> &indices) const {
	return getValue(flattenIndex(indices));
}

ValueType Tensor::getValue(const size_t index) const {
	if (isGpu) {
		return tensor_gpu::getValueAt(gpu_data, index);
	}

	return cpu_data[index];
}

void Tensor::setValue(const size_t indices, const ValueType value) {
	if (isGpu) {
		tensor_gpu::setValueAt(gpu_data, indices, value);
	} else {
		cpu_data[indices] = value;
	}
}

void Tensor::setValue(const std::vector<size_t> &indices,
                      const ValueType value) {
	setValue(flattenIndex(indices), value);
}

void Tensor::insertRange(const Tensor &other, const size_t startO,
                         const size_t startT, const size_t length) {
	if (isGpu) {
		tensor_gpu::copyDeviceToDevice(gpu_data + startT,
		                               other.gpu_data + startO,
		                               length * sizeof(ValueType));
	} else {
		for (size_t i = 0; i < length; ++i) {
			cpu_data[i + startT] = other.cpu_data[i + startO];
		}
	}
}

void Tensor::getData(std::vector<ValueType> &dest) const {
	if (isGpu) {
		// GPU mode: copy from device to host memory
		tensor_gpu::copyToHost(dest.data(), gpu_data, gpu_data_size * sizeof(ValueType));
	} else {
		// CPU mode: simple vector assignment
		dest = cpu_data;
	}
}

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

void Tensor::zero() {
	if (isGpu) {
		// GPU mode: use optimized zero kernel
		tensor_gpu::zero(gpu_data, gpu_data_size);
	} else {
		// CPU mode: use fill method
		fill(0);
	}
}

size_t Tensor::numElements() const {
	if (isGpu) {
		return gpu_data_size;
	}

	return cpu_data.size();
}

void Tensor::flatten() {
	shape = {numElements()};
	computeStrides();
}

void Tensor::setShape(const std::vector<size_t> &newShape) {
	shape = newShape;
	computeStrides();
}

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

void Tensor::toGpu() {
	if (isGpu)
		return;

	if (tensorCount > 0)
		throw std::runtime_error("Cannot switch to GPU mode: tensors already exist in CPU mode.");

	isGpu = true;
}

void Tensor::toCpu() {
	if (!isGpu)
		return;

	if (tensorCount > 0)
		throw std::runtime_error("Cannot switch to CPU mode: tensors already exist in GPU mode.");

	isGpu = false;
}
} // namespace nn::global
