#include "tensor_gpu.hpp"
#include <algorithm>
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
	}

	ValueType *newV = nullptr;
	tensor_gpu::copyToHost(newV, gpu_data, gpu_data_size * sizeof(ValueType));

	std::copy(newV, newV + gpu_data_size, dest.begin());
}

void Tensor::fill(const ValueType &value) {
	if (isGpu) {
		tensor_gpu::zero(gpu_data, gpu_data_size);
		tensor_gpu::add(gpu_data, value, gpu_data, gpu_data_size);
	} else {
		for (auto &n : cpu_data) {
			n = value;
		}
	}
}

Tensor &Tensor::operator=(const Tensor &other) {
	if (this == &other)
		return *this;

	shape = other.shape;
	strides = other.strides;
	if (!isGpu) {
		cpu_data = other.cpu_data;
	} else {
		ValueType *temp = (ValueType *)tensor_gpu::allocate(other.gpu_data_size * sizeof(ValueType));
		gpu_data_size = other.gpu_data_size;
		tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
		tensor_gpu::deallocate(gpu_data);
		gpu_data = temp;
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
	// CPU version, same as before
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
		tensor_gpu::add(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
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
		tensor_gpu::subtraction(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
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
		tensor_gpu::multiply(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
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
		tensor_gpu::division(gpu_data, other.gpu_data, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x *= scalar;
	} else {
		tensor_gpu::multiply(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator-=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x -= scalar;
	} else {
		tensor_gpu::subtraction(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator+=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x += scalar;
	} else {
		tensor_gpu::add(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor &Tensor::operator/=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x /= scalar;
	} else {
		tensor_gpu::division(gpu_data, scalar, gpu_data, gpu_data_size);
	}
	return *this;
}

Tensor Tensor::matmul(const Tensor &other) const {
	const auto &aShape = shape;
	const auto &bShape = other.shape;
	if (aShape.size() != 2 || bShape.size() != 1)
		throw std::runtime_error("matmul: unsupported shapes.");

	size_t M = aShape[0];
	size_t K = aShape[1];
	if (K != bShape[0])
		throw std::runtime_error("matmul: shape mismatch.");
	Tensor result({M});

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
		return result;
	}
	tensor_gpu::matmul(gpu_data, other.gpu_data, result.gpu_data, M, K);
	return result;
}

Tensor Tensor::outer(const Tensor &a, const Tensor &b) {
	if (a.shape.size() != 1 || b.shape.size() != 1) {
		throw std::runtime_error("outer: both tensors must be 1D vectors");
	}

	size_t m = a.shape[0];
	size_t n = b.shape[0];

	Tensor result({m, n});

	if (!isGpu) {
		float *r = result.cpu_data.data();
		const float *A = a.cpu_data.data();
		const float *B = b.cpu_data.data();

		for (size_t i = 0; i < m; ++i) {
			for (size_t j = 0; j < n; ++j) {
				r[i * n + j] = A[i] * B[j];
			}
		}
		return result;
	}
	tensor_gpu::outer(a.gpu_data, b.gpu_data, result.gpu_data, m, n);
	return result;
}

Tensor Tensor::matmulT(const Tensor &vec) const {
	const auto &wShape = shape;
	const auto &vShape = vec.shape;

	if (wShape.size() != 2 || vShape.size() != 1)
		throw std::runtime_error("matmulT: bad dimensions");

	size_t M = wShape[0];
	size_t N = wShape[1];
	if (vShape[0] != M)
		throw std::runtime_error("matmulT: incompatible");

	Tensor result({N}, 0.0f);

	if (!isGpu) {
		const float *W = cpu_data.data();
		const float *V = vec.cpu_data.data();
		float *R = result.cpu_data.data();

		for (size_t i = 0; i < N; ++i) {
			float sum = 0.0f;
			for (size_t j = 0; j < M; ++j) {
				sum += W[j * N + i] * V[j];
			}
			R[i] = sum;
		}
		return result;
	}
	tensor_gpu::matmulT(gpu_data, vec.gpu_data, result.gpu_data, M, N);
	return result;
}

Tensor::~Tensor() {
	if (isGpu) {
		tensor_gpu::deallocate(gpu_data);
	}
}
} // namespace nn::global
