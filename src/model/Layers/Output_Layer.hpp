#ifndef OUTPUT_LAYER_HPP
#define OUTPUT_LAYER_HPP

#include "layer.hpp"

class Output_Layer : public Layer {
  public:
	Output_Layer(int _size, int _prev_size, const double init_value) : Layer(_size, _prev_size, init_value) {}
	Output_Layer(Layer const &other) : Layer(other) {}
	void forward(const vector<double> &metrix) override;
	LayerType getType() const override { return LayerType::OUTPUT; }
};

#endif // OUTPUT_LAYER_HPP
