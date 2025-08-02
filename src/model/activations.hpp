#ifndef ACTIVATIONSP
#define ACTIVATIONSP

#include "tensor.hpp"
#include <Globals.hpp>
#include <cmath>

namespace nn::model {
constexpr global::ValueType RELU_LEAKY_ALPHA = 0.01;

constexpr global::ValueType maxValue(const global::ValueType &a, const float &b) {
	return (a < b) ? b : a;
}

enum class ActivationType {
	Relu,
	LeakyRelu,
	Sigmoid,
	Tanh,
	Softmax,
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

	static void relu(const global::Tensor &net, global::Tensor &out);
	static void derivativeRelu(const global::Tensor &net,
	                           global::Tensor &out);

	static void leakyRelu(const global::Tensor &net,
	                      global::Tensor &out);
	static void derivativeLeakyRelu(const global::Tensor &net,
	                                global::Tensor &out);

	static void sigmoid(const global::Tensor &net,
	                    global::Tensor &out);
	static void derivativeSigmoid(const global::Tensor &net,
	                              global::Tensor &out);

	static void tanh(const global::Tensor &net, global::Tensor &out);
	static void derivativeTanh(const global::Tensor &net,
	                           global::Tensor &out);

	static void softmax(const global::Tensor &net,
	                    global::Tensor &out);

	static global::ValueType maxVector(const global::Tensor &metrix);

  public:
	Activation(const ActivationType activationType_)
	    : activationType(activationType_) {}
	Activation(const Activation &other)
	    : activationType(other.activationType) {}
	~Activation() = default;

	global::ValueType activate(const global::ValueType x) const;
	global::ValueType derivativeActivate(const global::ValueType x) const;

	void activate(const global::Tensor &net,
	              global::Tensor &out) const;
	void derivativeActivate(const global::Tensor &net,
	                        global::Tensor &out) const;

	ActivationType getType() { return activationType; }
};
} // namespace nn::model

#endif // ACTIVATIONS
