#ifndef ACTIVATIONSP
#define ACTIVATIONSP

#include "neuron.hpp"
#include <cmath>
#include <vector>

#define RELU_LEAKY_ALPHA 0.01

enum class activation {
	relu_ = 0,
	leaky_relu_ = 1,
	sigmoid_ = 2,
	tanh_ = 3,
	none = 4,
};

class activations {
  private:
	const activation _activation;
	inline double Relu(const double x) const;
	inline double DerivativeRelu(const double x) const;
	inline double LeakyRelu(const double x) const;
	inline double DerivativeLeakyRelu(const double x) const;
	inline double Sigmoid(const double z) const;
	inline double DerivativeSigmoid(const double z) const;
	inline double Tanh(const double z) const;
	inline double DerivativeTanh(const double z) const;
	static double max_vector(const std::vector<double> &metrix);

  public:
	activations(const activation activation) : _activation(activation) {}
	activations(const activations &other) : _activation(other._activation) {}
	double activate(const double x) const;
	double DerivativeActivate(const double x) const;
	static void Softmax(neurons &metrix);
};

#endif // ACTIVATIONS
