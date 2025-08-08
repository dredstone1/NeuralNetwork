#ifndef TENSOR
#define TENSOR

#include "../src/model/tensor_gpu.hpp"
#include <vector>

namespace nn::model {
class Activation;
}

namespace nn::global {

class Tensor {
  private:
	std::vector<ValueType> cpu_data;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	ValueType *gpu_data = nullptr;
	std::size_t gpu_data_size{0};

	static const bool isGpu{true};

	void computeStrides();
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

	void loadTempGpu() const;

	friend model::Activation;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, float init = 0.0f);
	Tensor(const Tensor &other);

	~Tensor();

	Tensor &operator=(const Tensor &other);

	ValueType getValue(const std::vector<size_t> &newShape) const;
	void setValue(const std::vector<size_t> &newShape, const ValueType value);

	// Shape and size
	size_t numElements() const;
	const std::vector<size_t> &getShape() const { return shape; }
	const std::vector<size_t> &getStrides() const { return strides; }
	void getData(std::vector<ValueType> &dest) const;
	void fill(const ValueType &value);

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
	void matmulT(const Tensor &vec, Tensor &result) const;
};
} // namespace nn::global

#endif // TENSOR
