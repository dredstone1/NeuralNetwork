#ifndef ACTIVATIONSP
#define ACTIVATIONSP

#include "neuron.hpp"
#include <cmath>

namespace nn::model {
constexpr global::ValueType RELU_LEAKY_ALPHA = 0.01;

enum class ActivationType {
	Relu,
	LeakyRelu,
	Sigmoid,
	Tanh,
	None,
};

class Activation {
  private:
	const ActivationType activationType;
	static global::ValueType relu(const global::ValueType x);
	static global::ValueType derivativeRelu(const global::ValueType x);
	static global::ValueType leakyRelu(const global::ValueType x);
	static global::ValueType derivativeLeakyRelu(const global::ValueType x);
	static global::ValueType sigmoid(const global::ValueType z);
	static global::ValueType derivativeSigmoid(const global::ValueType z);
	static global::ValueType tanh(const global::ValueType z);
	static global::ValueType derivativeTanh(const global::ValueType z);
	static global::ValueType maxVector(const global::ParamMetrix &metrix);

  public:
	Activation(const ActivationType activationType_)
	    : activationType(activationType_) {}
	Activation(const Activation &other)
	    : activationType(other.activationType) {}
	global::ValueType activate(const global::ValueType x) const;
	global::ValueType derivativeActivate(const global::ValueType x) const;
	static void softmax(Neurons &metrix);
};
} // namespace nn::model

#endif // ACTIVATIONS
