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
	size_t flattenIndex(const std::vector<size_t> &indices) const;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, float init = 0.0f);

	// Element access
	ValueType &operator()(const std::vector<size_t> &indices);
	ValueType operator()(const std::vector<size_t> &indices) const;

	// Shape and size
	const std::vector<size_t> &getShape() const;
	size_t numElements() const;
	const std::vector<ValueType> &getData() const { return data; }

	// Iterators (for range-based loops)
	auto begin() noexcept { return data.begin(); }
	auto end() noexcept { return data.end(); }
	auto begin() const noexcept { return data.begin(); }
	auto end() const noexcept { return data.end(); }

	// Arithmetic operations
	Tensor operator+(const Tensor &other) const;
	Tensor operator*(const Tensor &other) const;

	Tensor &operator+=(const Tensor &other);
	Tensor &operator-=(const Tensor &other);
	Tensor &operator*=(const Tensor &other);
	Tensor &operator/=(const Tensor &other);

	// Scalar operations
	Tensor &operator*=(ValueType scalar);
	Tensor &operator/=(ValueType scalar);
	Tensor operator*(ValueType scalar) const;

	Tensor matmul(const Tensor &other) const;
	static Tensor outer(const Tensor &a, const Tensor &b);
	Tensor matmulT(const Tensor &vec) const;
};

} // namespace nn::global

#endif // TENSOR
