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
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		result.data[i] = data[i] + other.data[i];
	return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
	if (shape != other.shape)
		throw std::invalid_argument("Shape mismatch in Tensor::operator+=.");
	const size_t N = data.size();
	for (size_t i = 0; i < N; ++i)
		data[i] += other.data[i];
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

Tensor Tensor::operator*(ValueType scalar) const {
	Tensor result(*this);
	result *= scalar;
	return result;
}

} // namespace nn::global
