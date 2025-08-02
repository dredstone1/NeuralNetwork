#include "../../include/tensor.hpp"

namespace nn::global {
Tensor::Tensor(const std::vector<size_t> &shape, float init)
    : shape(shape) {
	// Compute total size
	size_t totalSize = 1;
	for (size_t s : shape)
		totalSize *= s;
	data.resize(totalSize, init);

	// Compute strides
	strides.resize(shape.size());
	size_t stride = 1;
	for (int i = shape.size() - 1; i >= 0; --i) {
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

float &Tensor::operator()(const std::vector<size_t> &indices) {
	size_t flatIndex = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
		flatIndex += indices[i] * strides[i];
	}
	return data[flatIndex];
}

float Tensor::operator()(const std::vector<size_t> &indices) const {
	size_t flatIndex = 0;
	for (size_t i = 0; i < indices.size(); ++i) {
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
