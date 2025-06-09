#include "activations.hpp"
#include "Globals.hpp"

namespace nn::model {
global::ValueType Activation::activate(const global::ValueType x) const {
	switch (activationType) {
	case ActivationType::Relu:
		return relu(x);
	case ActivationType::LeakyRelu:
		return leakyRelu(x);
	case ActivationType::Sigmoid:
		return sigmoid(x);
	case ActivationType::Tanh:
		return tanh(x);
	case ActivationType::None:
		return x;
	}
	return x;
}

global::ValueType Activation::derivativeActivate(const global::ValueType x) const {
	switch (activationType) {
	case ActivationType::Relu:
		return derivativeRelu(x);
	case ActivationType::LeakyRelu:
		return derivativeLeakyRelu(x);
	case ActivationType::Sigmoid:
		return derivativeSigmoid(x);
	case ActivationType::Tanh:
		return derivativeTanh(x);
	case ActivationType::None:
		return x;
	}

	return x;
}

global::ValueType Activation::maxVector(const global::ParamMetrix &metrix) {
	global::ValueType max = metrix[0];
	for (auto &value : metrix) {
		if (value > max) {
			max = value;
		}
	}

	return max;
}

void Activation::softmax(Neurons &metrix) {
	global::ValueType max = maxVector(metrix.net);
	global::ValueType sum = 0.0;

	for (size_t i = 0; i < metrix.size(); i++) {
		global::ValueType x = metrix.net[i] - max;
		if (x < -700.0)
			x = -700.0;
		if (x > 700.0)
			x = 700.0;
		metrix.out[i] = std::exp(x);
		sum += metrix.out[i];
	}

	sum = std::max(sum, 1e-10);

	for (size_t i = 0; i < metrix.size(); i++) {
		metrix.out[i] /= sum;
	}
}

global::ValueType Activation::relu(const global::ValueType x) {
	return std::max(0.0, x);
}
global::ValueType Activation::derivativeRelu(const global::ValueType x) {
	return (x > 0) ? 1.0 : 0.0;
}

global::ValueType Activation::leakyRelu(const global::ValueType x) {
	return (x > 0) ? x : RELU_LEAKY_ALPHA * x;
}
global::ValueType Activation::derivativeLeakyRelu(const global::ValueType x) {
	return (x > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

global::ValueType Activation::sigmoid(const global::ValueType z) {
	return 1.0 / (1.0 + std::exp(-z));
}
global::ValueType Activation::derivativeSigmoid(const global::ValueType z) {
	global::ValueType s = sigmoid(z);
	return s * (1.0 - s);
}

global::ValueType Activation::tanh(const global::ValueType z) {
	return std::tanh(z);
}
global::ValueType Activation::derivativeTanh(const global::ValueType z) {
	global::ValueType t = std::tanh(z);
	return 1.0 - t * t;
}
} // namespace nn::model
