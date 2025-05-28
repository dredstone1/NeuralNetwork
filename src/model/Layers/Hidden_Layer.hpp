#ifndef HIDDEN_LAYER_HPP
#define HIDDEN_LAYER_HPP

#include "../activations.hpp"
#include "layer.hpp"

using namespace ActivationFunctions;

class Hidden_Layer : public Layer {
  private:
	ActivationFunction activate;

  public:
	Hidden_Layer(const int _size, const int _prev_size, const activations activation, const double init_value) : Layer(_size, _prev_size, init_value), activate(activation) {}
	Hidden_Layer(const Hidden_Layer &other) : Layer(other), activate(other.activate) {}
	void forward(const std::vector<double> &metrix) override;
	LayerType getType() const override { return LayerType::HIDDEN; }
	double activate_(const double x) const { return activate.activate(x); }
	double Derivative_activate(const double x) { return activate.DerivativeActivate(x); }
};

#endif // HIDDEN_LAYER_HPP
