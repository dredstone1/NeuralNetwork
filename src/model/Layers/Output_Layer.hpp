#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "layer.hpp"

namespace nn {
class Output_Layer : public Layer {
  public:
	Output_Layer(const int _size, const int _prev_size, const Global::ValueType init_value)
	    : Layer(_size, _prev_size, init_value) {}
	Output_Layer(const Layer &other)
	    : Layer(other) {}
	void forward(const std::vector<Global::ValueType> &metrix, const Global::ValueType) override;
	LayerType getType() const override { return LayerType::OUTPUT; }
};
} // namespace nn

#endif // OUTPUT_LAYER
