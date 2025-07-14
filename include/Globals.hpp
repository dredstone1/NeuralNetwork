#ifndef GLOBAL
#define GLOBAL

#include <cstdint>
#include <vector>

namespace nn::global {
using ValueType = float;
using ParamMetrix = std::vector<ValueType>;

using Transformation = global::ParamMetrix(const global::ParamMetrix &sample);

struct Prediction {
	size_t index;
	global::ValueType value;
	Prediction() : index(0), value(0) {}
	Prediction(const int index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
};

constexpr std::uint32_t NEURON_WIDTH = 40;
constexpr float MIN_NEURON_WIDTH = 6.0f;
constexpr float MAX_NEURON_WIDTH = NEURON_WIDTH;
constexpr float MIN_GAP = 2.0f;
} // namespace nn::global

#endif // GLOBAL
