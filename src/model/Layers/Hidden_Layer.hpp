#ifndef HIDDEN_LAYER
#define HIDDEN_LAYER

#include "../activations.hpp"
#include "layer.hpp"

namespace nn {
class Hidden_Layer : public Layer {
  private:
	activations activate;

  public:
	Hidden_Layer(const int _size, const int _prev_size, const activation activation, const Global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value),
	      activate(activation) {}
	Hidden_Layer(const Hidden_Layer &other)
	    : Layer(other),
	      activate(other.activate) {}
	void forward(const std::vector<Global::ValueType> &metrix) override;
	LayerType getType() const override { return LayerType::HIDDEN; }
	Global::ValueType activate_(const Global::ValueType x) const { return activate.activate(x); }
	Global::ValueType Derivative_activate(const Global::ValueType x) { return activate.DerivativeActivate(x); }
};
} // namespace nn

#endif // HIDDEN_LAYER
