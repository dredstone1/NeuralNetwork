#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "neuron.hpp"
#include <cmath>

using namespace std;

namespace ActivationFunctions {
#define RELU_LEAKY_ALPHA 0.01

// class Activations {
//   public:
//     virtual double activation(const double x) const = 0;
//     virtual double derivative(const double x) const = 0;
// };
//
// class Relu : public Activations {
//   public:
//     double activation(const double x) const override;
//     double derivative(const double x) const override;
// };
//
// class LeakyRelu : public Activations {
//   public:
//     double activation(const double x) const override;
//     double derivative(const double x) const override;
// };
//
// class Softmax {
//   private:
//     static double max_vector(const vector<double> &metrix);
//
//   public:
//     static void softmax(neurons &metrix);
// };

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
