#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "neuron.hpp"
#include <cmath>

using namespace std;

namespace ActivationFunctions {
#define RELU_LEAKY_ALPHA 0.01

enum activations {
	relu_,
	leaky_relu_,
	sigmoid_,
	tanh_,
    none,
};

struct ActivationFunction {
	const activations _activation;
	ActivationFunction(const activations activation) : _activation(activation) {}
	ActivationFunction(const ActivationFunction &other) : _activation(other._activation) {}
	double activate(const double x) const;
	double DerivativeActivate(const double x) const;
};

inline double Relu(const double x) {
	return max(0.0, x);
}
inline double DerivativeRelu(const double x) {
	return (x > 0) ? 1.0 : 0.0;
}

inline double LeakyRelu(const double x) {
	return (x > 0) ? x : RELU_LEAKY_ALPHA * x;
}
inline double DerivativeLeakyRelu(const double x) {
	return (x > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

inline double Sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}
inline double DerivativeSigmoid(const double x) {
	return x * (1.0 - x);
}

inline double Tanh(const double x) {
	return tanh(x);
}
inline double DerivativeTanh(const double x) {
	return 1.0 - x * x;
}

void Softmax(neurons &metrix);
} // namespace ActivationFunctions

#endif // ACTIVATIONS_HPP
