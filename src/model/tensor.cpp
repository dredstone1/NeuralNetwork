#include "../../include/tensor.hpp"
#include <initializer_list>
#include <numeric>
#include <stdexcept>

namespace nn::global {
Tensor::Tensor(const std::vector<size_t> &shape, float init)
    : shape(shape) {
	if (shape.empty()) {
		throw std::invalid_argument("Tensor shape cannot be empty.");
	}

	size_t totalSize = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
	data.resize(totalSize, init);
	computeStrides();
}

void Tensor::computeStrides() {
	strides.resize(shape.size());
	size_t stride = 1;
	for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
		strides[i] = stride;
		stride *= shape[i];
	}
}

const std::vector<size_t> &Tensor::getShape() const {
	return shape;
}

size_t Tensor::numElements() const {
	return data.size();
}

ValueType &Tensor::operator()(const std::vector<size_t> &indices) {
	if (indices.size() != shape.size()) {
		throw std::invalid_argument("Incorrect number of indices.");
	}

	size_t flatIndex = 0;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (indices[i] >= shape[i])
			throw std::out_of_range("Index out of bounds.");
		flatIndex += indices[i] * strides[i];
	}
	return data[flatIndex];
}

ValueType Tensor::operator()(const std::vector<size_t> &indices) const {
	if (indices.size() != shape.size()) {
		throw std::invalid_argument("Incorrect number of indices.");
	}

	size_t flatIndex = 0;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (indices[i] >= shape[i])
			throw std::out_of_range("Index out of bounds.");
		flatIndex += indices[i] * strides[i];
	}
	return data[flatIndex];
}

Tensor Tensor::operator+(const Tensor &other) const {
	Tensor result(shape);
	for (size_t i = 0; i < data.size(); ++i)
		result.data[i] = data[i] + other.data[i];
	return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
	if (shape != other.shape) {
		throw std::invalid_argument("Shape mismatch in Tensor::operator+=.");
	}
	for (size_t i = 0; i < data.size(); ++i) {
		data[i] += other.data[i];
	}
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
	if (shape != other.shape) {
		throw std::invalid_argument("Shape mismatch in Tensor::operator-=.");
	}
	for (size_t i = 0; i < data.size(); ++i) {
		data[i] -= other.data[i];
	}
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
	if (shape != other.shape) {
		throw std::invalid_argument("Shape mismatch in Tensor::operator*=.");
	}
	for (size_t i = 0; i < data.size(); ++i) {
		data[i] *= other.data[i];
	}
	return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
	Tensor result = *this;
	result *= other;
	return result;
}

Tensor &Tensor::operator*=(ValueType scalar) {
	for (auto &x : data) {
		x *= scalar;
	}
	return *this;
}

Tensor Tensor::operator*(ValueType scalar) const {
	Tensor result = *this;
	result *= scalar;
	return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
	size_t M = shape[0];
	size_t K = shape[1];
	size_t N = other.shape[1];

	Tensor result({M, N}, 0.0f);

	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += (*this)({i, k}) * other({k, j});
			}
			result({i, j}) = sum;
		}
	}

	return result;
}
} // namespace nn::global
