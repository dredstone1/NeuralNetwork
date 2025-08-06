#ifndef TENSOR
#define TENSOR

#include "../src/model/tensor_gpu.hpp"
#include <cstddef>
#include <vector>

namespace nn::model {
class Activation;
}

namespace nn::global {

class Tensor {
  private:
	std::vector<ValueType> cpu_data;
	std::vector<size_t> cpu_shape;
	std::vector<size_t> cpu_strides;

	ValueType *gpu_data = nullptr;
	std::size_t gpu_data_size{0};
	size_t *gpu_shape = nullptr;
	size_t *gpu_strides = nullptr;
	size_t gpu_shape_size{0};

	static const bool isGpu{true};

	void computeStrides();
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

	void loadTempGpu() const;

	friend model::Activation;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, float init = 0.0f);
	Tensor(const Tensor &other);

	Tensor &operator=(const Tensor &other);

	// Element access
	ValueType &operator()(const std::vector<size_t> &indices);
	ValueType operator()(const std::vector<size_t> &indices) const;
	ValueType &operator[](size_t i);
	const ValueType &operator[](size_t i) const;

	// Iterators (for range-based loops)
	auto begin() noexcept { return cpu_data.begin(); }
	auto end() noexcept { return cpu_data.end(); }
	auto begin() const noexcept { return cpu_data.begin(); }
	auto end() const noexcept { return cpu_data.end(); }

	// Shape and size
	size_t numElements() const;
	void getData(std::vector<ValueType> &dest) const;
	void fill(const ValueType &value);

	// Arithmetic operations
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
