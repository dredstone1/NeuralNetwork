#include "tensor_gpu.hpp"
#include <numeric>
#include <stdexcept>
#include <tensor.hpp>
#include <vector>

namespace nn::global {
Tensor::Tensor(const std::vector<size_t> &shape_, float init) {
	if (shape_.empty()) {
		throw std::invalid_argument("Tensor shape cannot be empty.");
	}

	size_t totalSize = std::accumulate(
	    shape_.begin(),
	    shape_.end(),
	    size_t(1),
	    std::multiplies<>());

	shape = shape_;
	if (!isGpu) {
		cpu_data.assign(totalSize, init);
	} else {
		gpu_data = (ValueType *)tensor_gpu::allocate(totalSize * sizeof(ValueType));
		gpu_data_size = totalSize;
		fill(init);
	}

	computeStrides();
}

Tensor::Tensor(const Tensor &other) {
	shape = other.shape;
	strides = other.strides;
	if (isGpu) {
		gpu_data_size = other.gpu_data_size;
		gpu_data = (ValueType *)tensor_gpu::allocate(gpu_data_size * sizeof(ValueType));
		tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
	} else {
		cpu_data = other.cpu_data;
	}
}

size_t Tensor::numElements() const {
	if (isGpu) {
		return gpu_data_size;
	}
	return cpu_data.size();
}

void Tensor::getData(std::vector<ValueType> &dest) const {
	if (!isGpu) {
		dest = cpu_data;
	} else {
		tensor_gpu::copyToHost(dest.data(), gpu_data, gpu_data_size * sizeof(ValueType));
	}
}

void Tensor::fill(const ValueType &value) {
	if (isGpu) {
		tensor_gpu::zero(gpu_data, gpu_data_size);
		tensor_gpu::add_scalar(gpu_data, value, gpu_data, gpu_data_size);
	} else {
		for (auto &n : cpu_data) {
			n = value;
		}
	}
}

void Tensor::zero() {
	if (isGpu) {
		tensor_gpu::zero(gpu_data, gpu_data_size);
	} else {
		fill(0);
	}
}

Tensor &Tensor::operator=(const Tensor &other) {
	if (this == &other)
		return *this;

	if (!isGpu) {
		cpu_data = other.cpu_data;
	} else {
		ValueType *temp = gpu_data;
		if (gpu_data_size != other.gpu_data_size) {
			temp = (ValueType *)tensor_gpu::allocate(other.gpu_data_size * sizeof(ValueType));

			gpu_data_size = other.gpu_data_size;
		}
		tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));

		if (gpu_data_size != other.gpu_data_size) {
			tensor_gpu::deallocate(gpu_data);
			gpu_data = temp;
		}
	}

	shape = other.shape;
	strides = other.strides;
	return *this;
}

Tensor &Tensor::operator=(const std::vector<ValueType> &other) {
	if (other.size() != numElements()) {
		throw std::invalid_argument("");
	}

	if (!isGpu) {
		cpu_data = other;
	} else {
		tensor_gpu::copyToDevice(gpu_data, other.data(), gpu_data_size * sizeof(ValueType));
	}

	return *this;
}

void Tensor::computeStrides() {
	const size_t dim = shape.size();
	strides.resize(dim);
	size_t stride = 1;
	for (size_t i = dim; i-- > 0;) {
		strides[i] = stride;
		stride *= shape[i];
	}
}

inline size_t Tensor::flattenIndex(const std::vector<size_t> &indices) const {
	if (indices.size() != shape.size()) {
		throw std::invalid_argument("Incorrect number of indices.");
	}
	size_t index = 0;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (indices[i] >= shape[i])
			throw std::out_of_range("Index out of bounds.");
		index += indices[i] * strides[i];
	}
	return index;
}

ValueType Tensor::getValue(const std::vector<size_t> &indices) const {
	if (!isGpu) {
		return cpu_data[flattenIndex(indices)];
	}

	return tensor_gpu::getValueAt(gpu_data, flattenIndex(indices));
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
	if (!isGpu) {
		cpu_data[flattenIndex(indices)] = value;
	} else {
		tensor_gpu::setValueAt(gpu_data, flattenIndex(indices), value);
	}
}

Tensor &Tensor::operator+=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator+=.");
	if (!isGpu) {
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] += other.cpu_data[i];
	} else {
		tensor_gpu::add_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator-=.");
	if (!isGpu) {
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] -= other.cpu_data[i];
	} else {
		tensor_gpu::subtraction_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator*=.");
	if (!isGpu) {
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] *= other.cpu_data[i];
	} else {
		tensor_gpu::multiply_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator/=.");
	if (!isGpu) {
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] /= other.cpu_data[i];
	} else {
		tensor_gpu::division_vec(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x *= scalar;
	} else {
		tensor_gpu::multiply_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator-=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x -= scalar;
	} else {
		tensor_gpu::subtraction_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator+=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x += scalar;
	} else {
		tensor_gpu::add_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator/=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x /= scalar;
	} else {
		tensor_gpu::division_scalar(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

void Tensor::matmul(const Tensor &other, Tensor &result) const {
	const auto &aShape = shape;
	const auto &bShape = other.shape;
	if (aShape.size() != 2 || bShape.size() != 1)
		throw std::runtime_error("matmul: unsupported shapes.");

	size_t M = aShape[0];
	size_t K = aShape[1];
	if (K != bShape[0])
		throw std::runtime_error("matmul: shape mismatch.");

	result.zero();

	if (!isGpu) {
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
	} else {
		tensor_gpu::matmul(gpu_data, other.gpu_data, result.gpu_data, M, K);
	}
}

void Tensor::outer(const Tensor &a, const Tensor &b, Tensor &result) {
	if (a.shape.size() != 1 || b.shape.size() != 1) {
		throw std::runtime_error("outer: both tensors must be 1D vectors");
	}

	size_t m = a.shape[0];
	size_t n = b.shape[0];

	result.zero();

	if (!isGpu) {
		float *r = result.cpu_data.data();
		const float *A = a.cpu_data.data();
		const float *B = b.cpu_data.data();

		for (size_t i = 0; i < m; ++i) {
			for (size_t j = 0; j < n; ++j) {
				r[i * n + j] += A[i] * B[j];
			}
		}
	} else {
		tensor_gpu::outer(a.gpu_data, b.gpu_data, result.gpu_data, m, n);
	}
}

void Tensor::matmulT(const Tensor &vec, Tensor &result) const {
	if (shape.size() != 2 || vec.shape.size() != 1)
		throw std::runtime_error("matmulT: bad dimensions");
	if (vec.shape[0] != shape[0])
		throw std::runtime_error("matmulT: incompatible");

	result.zero();

	if (!isGpu) {
		for (size_t i = 0; i < shape[1]; ++i) {
			for (size_t j = 0; j < shape[0]; ++j) {
				result.cpu_data[i] += cpu_data[j * shape[1] + i] * vec.cpu_data[j];
			}
		}
	} else {
		tensor_gpu::matmulT(gpu_data, vec.gpu_data, result.gpu_data, shape[0], shape[1]);
	}
}

Tensor::~Tensor() {
	if (isGpu) {
		tensor_gpu::deallocate(gpu_data);
	}
}
} // namespace nn::global
