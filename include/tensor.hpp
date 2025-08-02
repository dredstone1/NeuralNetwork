#ifndef TENSOR
#define TENSOR

#include <cstddef>
#include <vector>

namespace nn::global {
using ValueType = float;

class Tensor {
  private:
	std::vector<ValueType> data;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	void computeStrides();

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, const float init = 0.0f);

	// Access
	ValueType &operator()(const std::vector<size_t> &indices);
	ValueType operator()(const std::vector<size_t> &indices) const;

	// Utilities
	const std::vector<size_t> &getShape() const;
	size_t numElements() const;

	// Iterators for range-based loops
	auto begin() { return data.begin(); }
	auto end() { return data.end(); }
	auto begin() const { return data.begin(); }
	auto end() const { return data.end(); }

	// Math ops
	Tensor operator+(const Tensor &other) const;
	Tensor &operator+=(const Tensor &other);
	Tensor &operator-=(const Tensor &other);
	Tensor &operator*=(const Tensor &other);
	Tensor operator*(const Tensor &other) const;
	Tensor &operator*=(ValueType scalar);
	Tensor operator*(ValueType scalar) const;

	Tensor matmul(const Tensor &other) const;
};
} // namespace nn::global

#endif // TENSOR
