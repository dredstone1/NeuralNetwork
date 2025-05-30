#include "activations.hpp"

double activations::activate(const double x) const {
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

double activations::DerivativeActivate(const double x) const {
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

double activations::max_vector(const std::vector<double> &metrix) {
	double max = metrix[0];
	for (auto &value : metrix) {
		if (value > max) {
			max = value;
		}
	}
	return max;
}

void activations::Softmax(neurons &metrix) {
	double max = max_vector(metrix.net);
	double sum = 0.0;
	for (int i = 0; i < metrix.size(); ++i) {
		metrix.out[i] = exp(metrix.net[i] - max);
		sum += metrix.out[i];
	}
	for (int i = 0; i < metrix.size(); ++i) {
		metrix.out[i] /= sum;
	}
}

inline double activations::Relu(const double x) const {
	return std::max(0.0, x);
}
inline double activations::DerivativeRelu(const double x) const {
	return (x > 0) ? 1.0 : 0.0;
}

inline double activations::LeakyRelu(const double x) const {
	return (x > 0) ? x : RELU_LEAKY_ALPHA * x;
}
inline double activations::DerivativeLeakyRelu(const double x) const {
	return (x > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

inline double activations::Sigmoid(const double z) const {
	return 1.0 / (1.0 + std::exp(-z));
}
inline double activations::DerivativeSigmoid(const double z) const {
	double s = Sigmoid(z);
	return s * (1.0 - s);
}

inline double activations::Tanh(const double z) const {
	return std::tanh(z);
}
inline double activations::DerivativeTanh(const double z) const {
	double t = std::tanh(z);
	return 1.0 - t * t;
}
