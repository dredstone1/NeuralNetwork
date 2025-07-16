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

void Activation::activate(const global::ParamMetrix &net, global::ParamMetrix &out) const {
	switch (activationType) {
	case ActivationType::Relu:
		relu(net, out);
		break;
	case ActivationType::LeakyRelu:
		leakyRelu(net, out);
		break;
	case ActivationType::Sigmoid:
		sigmoid(net, out);
		break;
	case ActivationType::Tanh:
		tanh(net, out);
		break;
	case ActivationType::Softmax:
		softmax(net, out);
		break;
	default:
		break;
	}
}

void Activation::derivativeActivate(const global::ParamMetrix &net, global::ParamMetrix &out) const {
	switch (activationType) {
	case ActivationType::Relu:
		derivativeRelu(net, out);
		break;
	case ActivationType::LeakyRelu:
		derivativeLeakyRelu(net, out);
		break;
	case ActivationType::Sigmoid:
		derivativeSigmoid(net, out);
		break;
	case ActivationType::Tanh:
		derivativeTanh(net, out);
		break;
	default:
		break;
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

void Activation::relu(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] = relu(net[i]);
}

void Activation::derivativeRelu(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] *= derivativeRelu(net[i]);
}

void Activation::leakyRelu(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] = leakyRelu(net[i]);
}

void Activation::derivativeLeakyRelu(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] *= derivativeLeakyRelu(net[i]);
}

void Activation::sigmoid(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] = sigmoid(net[i]);
}

void Activation::derivativeSigmoid(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] *= derivativeSigmoid(net[i]);
}

void Activation::tanh(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] = tanh(net[i]);
}

void Activation::derivativeTanh(const global::ParamMetrix &net, global::ParamMetrix &out) {
	for (size_t i = 0; i < net.size(); ++i)
		out[i] *= derivativeTanh(net[i]);
}

void Activation::softmax(const global::ParamMetrix &net, global::ParamMetrix &out) {
	global::ValueType max = maxVector(net);
	global::ValueType sum = 0.0;

	for (size_t i = 0; i < net.size(); ++i) {
		global::ValueType x = net[i] - max;
		if (x < -700.0)
			x = -700.0;
		if (x > 700.0)
			x = 700.0;
		out[i] = std::exp(x);
		sum += out[i];
	}

	sum = maxValue(sum, 1e-10);

	for (size_t i = 0; i < out.size(); ++i) {
		out[i] /= sum;
	}
}
} // namespace nn::model
