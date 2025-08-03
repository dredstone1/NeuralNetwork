#include "../../include/tensor.hpp"
#include <numeric>
#include <stdexcept>

namespace nn::global {
Tensor::Tensor(const std::vector<size_t> &shape, float init)
    : shape(shape) {
	if (shape.empty()) {
		throw std::invalid_argument("Tensor shape cannot be empty.");
	}

	size_t totalSize = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
	data.assign(totalSize, init);
	computeStrides();
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

ValueType &Tensor::operator()(const std::vector<size_t> &indices) {
	return data[flattenIndex(indices)];
}

ValueType Tensor::operator()(const std::vector<size_t> &indices) const {
	return data[flattenIndex(indices)];
}

const std::vector<size_t> &Tensor::getShape() const {
	return shape;
}

size_t Tensor::numElements() const {
	return data.size();
}

Tensor Tensor::operator+(const Tensor &other) const {
	if (shape != other.shape) {
		throw std::invalid_argument("Shape mismatch in Tensor::operator+.");
	}
	Tensor result(shape);
	const float *a = data.data();
	const float *b = other.data.data();
	float *r = result.data.data();
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		r[i] = a[i] + b[i];
	return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch.");
	float *a = data.data();
	const float *b = other.data.data();
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		a[i] += b[i];
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator-=.");
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		data[i] -= other.data[i];
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator*=.");
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		data[i] *= other.data[i];
	return *this;
}

Tensor &Tensor::operator/=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator/=.");
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		data[i] /= other.data[i];
	return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
	Tensor result(*this);
	result *= other;
	return result;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	for (auto &x : data)
		x *= scalar;
	return *this;
}

Tensor &Tensor::operator/=(ValueType scalar) {
	for (auto &x : data)
		x /= scalar;
	return *this;
}

Tensor Tensor::operator*(ValueType scalar) const {
	Tensor result(*this);
	result *= scalar;
	return result;
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

	Tensor result({M}, 0.0f);

	const float *A = data.data();
	const float *B = other.data.data();
	float *R = result.data.data();

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

Tensor Tensor::outer(const Tensor &a, const Tensor &b) {
	const std::vector<size_t> &aShape = a.getShape();
	const std::vector<size_t> &bShape = b.getShape();

	if (aShape.size() != 1 || bShape.size() != 1) {
		throw std::runtime_error("outer: both tensors must be 1D vectors");
	}

	size_t m = aShape[0];
	size_t n = bShape[0];

	Tensor result({m, n}, 0.0f);

	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			result({i, j}) = a({i}) * b({j});
		}
	}

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

	const float* W = data.data();
	const float* V = vec.data.data();
	float* R = result.data.data();

	for (size_t i = 0; i < N; ++i) {
		float sum = 0.0f;
		for (size_t j = 0; j < M; ++j) {
			sum += W[j * N + i] * V[j];
		}
		R[i] = sum;
	}
	return result;
}
} // namespace nn::global
