#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "layer.hpp"

class Output_Layer : public Layer {
  public:
	Output_Layer(const int _size, const int _prev_size, const double init_value)
	    : Layer(_size, _prev_size, init_value) {}
	Output_Layer(const Layer &other)
	    : Layer(other) {}
	void forward(const std::vector<double> &metrix) override;
	LayerType getType() const override { return LayerType::OUTPUT; }
};

#endif // OUTPUT_LAYER
