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
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, float init = 0.0f);
	Tensor(const Tensor &other)
	    : data(other.data),
	      shape(other.shape),
	      strides(other.strides) {}

	Tensor &operator=(const Tensor &other);

	// Element access
	ValueType &operator()(const std::vector<size_t> &indices);
	ValueType operator()(const std::vector<size_t> &indices) const;
	inline ValueType &operator[](size_t i) { return data[i]; }
	inline const ValueType &operator[](size_t i) const { return data[i]; }

	// Iterators (for range-based loops)
	auto begin() noexcept { return data.begin(); }
	auto end() noexcept { return data.end(); }
	auto begin() const noexcept { return data.begin(); }
	auto end() const noexcept { return data.end(); }

	// Shape and size
	inline const std::vector<size_t> &getShape() const { return shape; }
	inline size_t numElements() const { return data.size(); }
	inline const std::vector<ValueType> &getData() const { return data; }
	inline void fill(const ValueType &value) { std::fill(begin(), end(), value); }

	// Arithmetic operations
	Tensor operator+(const Tensor &other) const;
	Tensor operator*(const Tensor &other) const;
	Tensor operator-(const Tensor &other) const;
	Tensor operator/(const Tensor &other) const;

	Tensor operator*(ValueType scalar) const;
	Tensor operator+(ValueType scalar) const;
	Tensor operator/(ValueType scalar) const;
	Tensor operator-(ValueType scalar) const;

	Tensor &operator+=(const Tensor &other);
	Tensor &operator-=(const Tensor &other);
	Tensor &operator*=(const Tensor &other);
	Tensor &operator/=(const Tensor &other);

	Tensor &operator*=(ValueType scalar);
	Tensor &operator/=(ValueType scalar);
	Tensor &operator+=(ValueType scalar);
	Tensor &operator-=(ValueType scalar);

	Tensor matmul(const Tensor &other) const;
	static Tensor outer(const Tensor &a, const Tensor &b);
	Tensor matmulT(const Tensor &vec) const;
};
} // namespace nn::global

#endif // TENSOR
