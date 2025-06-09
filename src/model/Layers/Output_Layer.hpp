#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "Globals.hpp"
#include "layer.hpp"

namespace nn::model {
class Output_Layer : public Layer {
  public:
	Output_Layer(const int _size, const int _prev_size, const global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value) {}
	Output_Layer(const Layer &other)
	    : Layer(other) {}
	void forward(const global::ParamMetrix &metrix) override;
	LayerType getType() const override { return LayerType::OUTPUT; }
};
} // namespace nn::model

#endif // OUTPUT_LAYER
