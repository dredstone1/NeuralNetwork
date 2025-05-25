#include "activations.hpp"

namespace ActivationFunctions {
double ActivationFunction::activate(const double x) const {
	switch (_activation) {
	case relu_:
		return Relu(x);
	case leaky_relu_:
		return LeakyRelu(x);
	case sigmoid_:
		return Sigmoid(x);
	case tanh_:
		return Tanh(x);
	case none:
		return x;
	}
	return x;
}

double ActivationFunction::DerivativeActivate(const double x) const {
	switch (_activation) {
	case relu_:
		return DerivativeRelu(x);
	case leaky_relu_:
		return DerivativeLeakyRelu(x);
	case sigmoid_:
		return DerivativeSigmoid(x);
	case tanh_:
		return DerivativeTanh(x);
	case none:
		return x;
	}
	return x;
}

static double max_vector(const vector<double> &metrix) {
	double max = metrix[0];
	for (auto &value : metrix) {
		if (value > max) {
			max = value;
		}
	}
	return max;
}

void Softmax(neurons &metrix) {
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
} // namespace ActivationFunctions
