#ifndef ACTIVATIONSP
#define ACTIVATIONSP

#include "neuron.hpp"
#include <cmath>

namespace nn {
constexpr Global::ValueType RELU_LEAKY_ALPHA = 0.01;

enum class activation {
	relu_,
	leaky_relu_,
	sigmoid_,
	tanh_,
	none,
};

class activations {
  private:
	const activation _activation;
	static Global::ValueType Relu(const Global::ValueType x);
	static Global::ValueType DerivativeRelu(const Global::ValueType x);
	static Global::ValueType LeakyRelu(const Global::ValueType x);
	static Global::ValueType DerivativeLeakyRelu(const Global::ValueType x);
	static Global::ValueType Sigmoid(const Global::ValueType z);
	static Global::ValueType DerivativeSigmoid(const Global::ValueType z);
	static Global::ValueType Tanh(const Global::ValueType z);
	static Global::ValueType DerivativeTanh(const Global::ValueType z);
	static Global::ValueType max_vector(const std::vector<Global::ValueType> &metrix);

  public:
	activations(const activation activation)
	    : _activation(activation) {}
	activations(const activations &other)
	    : _activation(other._activation) {}
	Global::ValueType activate(const Global::ValueType x) const;
	Global::ValueType DerivativeActivate(const Global::ValueType x) const;
	static void Softmax(neurons &metrix);
};
} // namespace nn

#endif // ACTIVATIONS
