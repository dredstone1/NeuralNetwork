#include "tensor_gpu.hpp"
#include <numeric>
#include <stdexcept>
#include <tensor.hpp>

namespace nn::global {
Tensor::Tensor(const std::vector<size_t> &shape, float init) {
	if (shape.empty()) {
		throw std::invalid_argument("Tensor shape cannot be empty.");
	}

	size_t totalSize = std::accumulate(
	    shape.begin(),
	    shape.end(),
	    size_t(1),
	    std::multiplies<>());

	if (!isGpu) {
		cpu_shape = shape;
		cpu_data.assign(totalSize, init);
	} else {
		gpu_shape = (size_t *)tensor_gpu::allocate(shape.size() * sizeof(size_t));
		tensor_gpu::copyToDevice(gpu_shape, shape.data(), gpu_data_size * sizeof(size_t));
		gpu_data = (ValueType *)tensor_gpu::allocate(totalSize * sizeof(ValueType));
		gpu_data_size = totalSize;
	}

	computeStrides();
}

ValueType &Tensor::operator[](size_t i) {
	if (isGpu) {
	}
	return cpu_data[i];
}

const ValueType &Tensor::operator[](size_t i) const {
	if (isGpu) {
	}
	return cpu_data[i];
}

size_t Tensor::numElements() const {
	if (isGpu) {
		return gpu_data_size;
	}
	return cpu_data.size();
}

const std::vector<ValueType> &Tensor::getData() const {
	if (!isGpu) {
		return cpu_data;
	}
}

void Tensor::fill(const ValueType &value) {
	std::fill(begin(), end(), value);
}

Tensor &Tensor::operator=(const Tensor &other) {
	if (this == &other)
		return *this;

	if (!isGpu) {
		cpu_data = other.cpu_data;
		cpu_shape = other.cpu_shape;
		cpu_strides = other.cpu_strides;
	} else {
		gpu_shape = (size_t *)tensor_gpu::allocate(other.gpu_shape_size * sizeof(size_t));
		gpu_data = (ValueType *)tensor_gpu::allocate(other.gpu_data_size * sizeof(ValueType));
		gpu_data_size = other.gpu_data_size;
		gpu_shape_size = other.gpu_shape_size;

		tensor_gpu::copyDeviceToDevice(gpu_data, other.gpu_data, gpu_data_size * sizeof(ValueType));
		tensor_gpu::copyDeviceToDevice(gpu_shape, other.gpu_shape, gpu_shape_size * sizeof(size_t));
		tensor_gpu::copyDeviceToDevice(gpu_strides, other.gpu_strides, gpu_shape_size * sizeof(size_t));
	}
	return *this;
}

void Tensor::computeStrides() {
	if (isGpu) {
		gpu_strides = (size_t *)tensor_gpu::allocate(gpu_shape_size * sizeof(size_t));
		tensor_gpu::computeStridesDevice(gpu_shape, gpu_strides, gpu_shape_size * sizeof(size_t));
	} else {
		const size_t dim = cpu_shape.size();
		cpu_strides.resize(dim);
		size_t stride = 1;
		for (size_t i = dim; i-- > 0;) {
			cpu_strides[i] = stride;
			stride *= cpu_shape[i];
		}
	}
}

inline size_t Tensor::flattenIndex(const std::vector<size_t> &indices) const {
	if (!isGpu) {
		if (indices.size() != cpu_shape.size()) {
			throw std::invalid_argument("Incorrect number of indices.");
		}
		size_t index = 0;
		for (size_t i = 0; i < cpu_shape.size(); ++i) {
			if (indices[i] >= cpu_shape[i])
				throw std::out_of_range("Index out of bounds.");
			index += indices[i] * cpu_strides[i];
		}
		return index;
	}
}

ValueType &Tensor::operator()(const std::vector<size_t> &indices) {
	if (!isGpu) {
		return cpu_data[flattenIndex(indices)];
	}
}

ValueType Tensor::operator()(const std::vector<size_t> &indices) const {
	if (!isGpu) {
		return cpu_data[flattenIndex(indices)];
	}
}

Tensor Tensor::operator+(const Tensor &other) const {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape) {
			throw std::invalid_argument("Shape mismatch in Tensor::operator+.");
		}
		Tensor result(cpu_shape);
		const float *a = cpu_data.data();
		const float *b = other.cpu_data.data();
		float *r = result.cpu_data.data();
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			r[i] = a[i] + b[i];
		return result;
	}
}

Tensor Tensor::operator-(const Tensor &other) const {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape) {
			throw std::invalid_argument("Shape mismatch in Tensor::operator-.");
		}
		Tensor result(cpu_shape);
		const float *a = cpu_data.data();
		const float *b = other.cpu_data.data();
		float *r = result.cpu_data.data();
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			r[i] = a[i] - b[i];
		return result;
	}
}

Tensor Tensor::operator/(const Tensor &other) const {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape) {
			throw std::invalid_argument("Shape mismatch in Tensor::operator/.");
		}
		Tensor result(cpu_shape);
		const float *a = cpu_data.data();
		const float *b = other.cpu_data.data();
		float *r = result.cpu_data.data();
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			r[i] = a[i] / b[i];
		return result;
	}
}

Tensor &Tensor::operator+=(const Tensor &other) {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape)
			throw std::invalid_argument("Shape mismatch in Tensor::operator+=.");
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] += other.cpu_data[i];
	} else {
	}
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape)
			throw std::invalid_argument("Shape mismatch in Tensor::operator-=.");
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] -= other.cpu_data[i];
	} else {
	}
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape)
			throw std::invalid_argument("Shape mismatch in Tensor::operator*=.");
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] *= other.cpu_data[i];
	} else {
	}
	return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
	if (!isGpu) {
		if (cpu_shape != other.cpu_shape)
			throw std::invalid_argument("Shape mismatch in Tensor::operator/=.");
		const size_t N = cpu_data.size();
		for (size_t i = 0; i < N; ++i)
			cpu_data[i] /= other.cpu_data[i];
	} else {
	}
	return *this;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x *= scalar;
	} else {
	}
	return *this;
}

Tensor &Tensor::operator-=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x -= scalar;
	} else {
	}
	return *this;
}

Tensor &Tensor::operator+=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x += scalar;
	} else {
	}
	return *this;
}

Tensor &Tensor::operator/=(ValueType scalar) {
	if (!isGpu) {
		for (auto &x : cpu_data)
			x /= scalar;
	} else {
	}
	return *this;
}

Tensor Tensor::operator*(ValueType scalar) const {
	Tensor result(*this);
	result *= scalar;
	return result;
}

Tensor Tensor::operator/(ValueType scalar) const {
	Tensor result(*this);
	result /= scalar;
	return result;
}

Tensor Tensor::operator-(ValueType scalar) const {
	Tensor result(*this);
	result -= scalar;
	return result;
}

Tensor Tensor::operator+(ValueType scalar) const {
	Tensor result(*this);
	result += scalar;
	return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
	if (!isGpu) {
		const auto &aShape = cpu_shape;
		const auto &bShape = other.cpu_shape;

		if (aShape.size() != 2 || bShape.size() != 1)
			throw std::runtime_error("matmul: unsupported shapes.");

		size_t M = aShape[0];
		size_t K = aShape[1];
		if (K != bShape[0])
			throw std::runtime_error("matmul: shape mismatch.");

		Tensor result({M});

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
}

Tensor Tensor::outer(const Tensor &a, const Tensor &b) {
	if (!isGpu) {
		if (a.cpu_shape.size() != 1 || b.cpu_shape.size() != 1) {
			throw std::runtime_error("outer: both tensors must be 1D vectors");
		}

		size_t m = a.cpu_shape[0];
		size_t n = b.cpu_shape[0];

		Tensor result({m, n});
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
}

Tensor Tensor::matmulT(const Tensor &vec) const {
	if (!isGpu) {
		const auto &wShape = cpu_shape;
		const auto &vShape = vec.cpu_shape;

		if (wShape.size() != 2 || vShape.size() != 1)
			throw std::runtime_error("matmulT: bad dimensions");

		size_t M = wShape[0];
		size_t N = wShape[1];
		if (vShape[0] != M)
			throw std::runtime_error("matmulT: incompatible");

		Tensor result({N}, 0.0f);

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
}
} // namespace nn::global
