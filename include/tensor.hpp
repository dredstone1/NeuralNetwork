#ifndef TENSOR
#define TENSOR

#include "../src/model/tensor_gpu.hpp"
#include <string>
#include <vector>

namespace nn::model {
class Activation;

namespace cnn {
class CNNetwork;
}
} // namespace nn::model

namespace nn::global {

class Tensor;
using Transformation = Tensor (*)(const Tensor &);

std::string shapeToString(const std::vector<size_t> &shape);
size_t computeTensorSize(const std::vector<size_t> &shape);

constexpr bool DEFAULT_GPU_MODE = false;
constexpr ValueType DEFAULT_INIT_VALUE = 0.0f;

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
	friend nn::global::Transformation;
	friend nn::model::cnn::CNNetwork;

  public:
	// Constructors
	Tensor(const std::vector<size_t> &shape, ValueType init = DEFAULT_INIT_VALUE);
	Tensor(const Tensor &other);

	~Tensor();

	Tensor &operator=(const Tensor &other);
	Tensor &operator=(const std::vector<ValueType> &other);

	ValueType getValue(const std::vector<size_t> &newShape) const;
	void setValue(const std::vector<size_t> &newShape, const ValueType value);

	ValueType getValue(const size_t newShape) const;
	void setValue(const size_t newShape, const ValueType value);

	void insertRange(const Tensor &other, const size_t startO,
	                 const size_t startT, const size_t length);

	// Shape and size
	size_t numElements() const;
	const std::vector<size_t> &getShape() const { return shape; }
	const std::vector<size_t> &getStrides() const { return strides; }
	void getData(std::vector<ValueType> &dest) const;
	void setData(const Tensor &other);
	void fill(const ValueType &value);
	void zero();

	void flatten();
	void setShape(const std::vector<size_t> &newShape);

	// Data access for testing
	ValueType *getGpuData() const { return gpu_data; }
	std::vector<ValueType> &getCpuData() { return cpu_data; }

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
	static bool getGpuState() { return isGpu; }
};

} // namespace nn::global

#endif // TENSOR
