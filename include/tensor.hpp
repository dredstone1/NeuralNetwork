#ifndef TENSOR
#define TENSOR

#include "../src/model/tensor_gpu.hpp"
#include <vector>

namespace nn::model {
class Activation;
void enableGpuMode();
} // namespace nn::model

namespace nn::global {

class Tensor {
  private:
	std::vector<ValueType> cpu_data;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	ValueType *gpu_data = nullptr;
	std::size_t gpu_data_size;

	static bool isGpu;
    static size_t tensorCount;

	void computeStrides();
	inline size_t flattenIndex(const std::vector<size_t> &indices) const;

	friend model::Activation;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, ValueType init = 0.0f);
	Tensor(const Tensor &other);

	~Tensor();

	Tensor &operator=(const Tensor &other);
	Tensor &operator=(const std::vector<ValueType> &other);

	ValueType getValue(const std::vector<size_t> &newShape) const;
	void setValue(const std::vector<size_t> &newShape, const ValueType value);
	void insertRange(const Tensor &other, const size_t startO,
	                 const size_t startT, const size_t length);

	// Shape and size
	size_t numElements() const;
	const std::vector<size_t> &getShape() const { return shape; }
	const std::vector<size_t> &getStrides() const { return strides; }
	void getData(std::vector<ValueType> &dest) const;
	void fill(const ValueType &value);
	void zero();

	Tensor &operator+=(const Tensor &other);
	Tensor &operator-=(const Tensor &other);
	Tensor &operator*=(const Tensor &other);
	Tensor &operator/=(const Tensor &other);

	Tensor &operator*=(ValueType scalar);
	Tensor &operator/=(ValueType scalar);
	Tensor &operator+=(ValueType scalar);
	Tensor &operator-=(ValueType scalar);

	void matmul(const Tensor &other, Tensor &result) const;
	static void outer(const Tensor &a, const Tensor &b, Tensor &result);
	void matmulT(const Tensor &vec, Tensor &result) const;

	static void toGpu();
	static void toCpu();
};

} // namespace nn::global

#endif // TENSOR
