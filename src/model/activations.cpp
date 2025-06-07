#include "activations.hpp"

namespace nn {
Global::ValueType activations::activate(const Global::ValueType x) const {
	switch (_activation) {
	case activation::relu_:
		return Relu(x);
	case activation::leaky_relu_:
		return LeakyRelu(x);
	case activation::sigmoid_:
		return Sigmoid(x);
	case activation::tanh_:
		return Tanh(x);
	case activation::none:
		return x;
	}
	return x;
}

Global::ValueType activations::DerivativeActivate(const Global::ValueType x) const {
	switch (_activation) {
	case activation::relu_:
		return DerivativeRelu(x);
	case activation::leaky_relu_:
		return DerivativeLeakyRelu(x);
	case activation::sigmoid_:
		return DerivativeSigmoid(x);
	case activation::tanh_:
		return DerivativeTanh(x);
	case activation::none:
		return x;
	}
	return x;
}

Global::ValueType activations::max_vector(const std::vector<Global::ValueType> &metrix) {
	Global::ValueType max = metrix[0];
	for (auto &value : metrix) {
		if (value > max) {
			max = value;
		}
	}
	return max;
}

void activations::Softmax(neurons &metrix) {
	Global::ValueType max = max_vector(metrix.net);
	Global::ValueType sum = 0.0;
	for (size_t i = 0; i < metrix.size(); ++i) {
		metrix.out[i] = exp(metrix.net[i] - max);
		sum += metrix.out[i];
	}
	for (size_t i = 0; i < metrix.size(); ++i) {
		metrix.out[i] /= sum;
	}
}

Global::ValueType activations::Relu(const Global::ValueType x) {
	return std::max(0.0, x);
}
Global::ValueType activations::DerivativeRelu(const Global::ValueType x) {
	return (x > 0) ? 1.0 : 0.0;
}

Global::ValueType activations::LeakyRelu(const Global::ValueType x) {
	return (x > 0) ? x : RELU_LEAKY_ALPHA * x;
}
Global::ValueType activations::DerivativeLeakyRelu(const Global::ValueType x) {
	return (x > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

Global::ValueType activations::Sigmoid(const Global::ValueType z) {
	return 1.0 / (1.0 + std::exp(-z));
}
Global::ValueType activations::DerivativeSigmoid(const Global::ValueType z) {
	Global::ValueType s = Sigmoid(z);
	return s * (1.0 - s);
}

Global::ValueType activations::Tanh(const Global::ValueType z) {
	return std::tanh(z);
}
Global::ValueType activations::DerivativeTanh(const Global::ValueType z) {
	Global::ValueType t = std::tanh(z);
	return 1.0 - t * t;
}
} // namespace nn
