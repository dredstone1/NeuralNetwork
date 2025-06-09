#ifndef HIDDEN_LAYER
#define HIDDEN_LAYER

#include "../activations.hpp"
#include "layer.hpp"

namespace nn::model {
class Hidden_Layer : public Layer {
  private:
	Activation activationFunction;

  public:
	Hidden_Layer(const int _size, const int _prev_size, const ActivationType activation, const global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value),
	      activationFunction(activation) {}
	Hidden_Layer(const Hidden_Layer &other)
	    : Layer(other),
	      activationFunction(other.activationFunction) {}

	void forward(const global::ParamMetrix &metrix) override;
	LayerType getType() const override { return LayerType::HIDDEN; }
	global::ValueType activation(const global::ValueType x) const { return activationFunction.activate(x); }
	global::ValueType derivativeActivation(const global::ValueType x) const { return activationFunction.derivativeActivate(x); }
};
} // namespace nn::model

#endif // HIDDEN_LAYER
