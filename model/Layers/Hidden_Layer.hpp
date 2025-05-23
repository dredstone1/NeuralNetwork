#ifndef HIDDEN_LAYER_HPP
#define HIDDEN_LAYER_HPP

#include "layer.hpp"

class Hidden_Layer : public Layer {
  public:
	Hidden_Layer(int _size, int _prev_size) : Layer(_size, _prev_size, true) {}
	Hidden_Layer(Layer const &other) : Layer(other) {}
	void forward(vector<double> const &metrix) override;
	LayerType getType() const override { return LayerType::HIDDEN; }
};

#endif // HIDDEN_LAYER_HPP
