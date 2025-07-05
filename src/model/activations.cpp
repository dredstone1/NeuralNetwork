#include "activations.hpp"

namespace nn::model {
global::ValueType Activation::activate(const global::ValueType z) const {
	switch (activationType) {
	case ActivationType::Relu:
		return relu(z);
	case ActivationType::LeakyRelu:
		return leakyRelu(z);
	case ActivationType::Sigmoid:
		return sigmoid(z);
	case ActivationType::Tanh:
		return tanh(z);
	default:
		return z;
	}
}

global::ValueType Activation::derivativeActivate(const global::ValueType z) const {
	switch (activationType) {
	case ActivationType::Relu:
		return derivativeRelu(z);
	case ActivationType::LeakyRelu:
		return derivativeLeakyRelu(z);
	case ActivationType::Sigmoid:
		return derivativeSigmoid(z);
	case ActivationType::Tanh:
		return derivativeTanh(z);
	default:
		return z;
	}
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

	for (size_t i = 0; i < metrix.size(); ++i) {
		global::ValueType x = metrix.net[i] - max;
		if (x < -700.0)
			x = -700.0;
		if (x > 700.0)
			x = 700.0;
		metrix.out[i] = std::exp(x);
		sum += metrix.out[i];
	}

	sum = maxValue(sum, 1e-10);

	for (size_t i = 0; i < metrix.size(); ++i) {
		metrix.out[i] /= sum;
	}
}

global::ValueType Activation::relu(const global::ValueType z) {
	return maxValue(z, 0.0f);
}
global::ValueType Activation::derivativeRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : 0.0;
}

global::ValueType Activation::leakyRelu(const global::ValueType z) {
	return (z > 0) ? z : RELU_LEAKY_ALPHA * z;
}
global::ValueType Activation::derivativeLeakyRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : RELU_LEAKY_ALPHA;
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
