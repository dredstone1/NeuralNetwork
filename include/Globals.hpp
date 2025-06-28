#ifndef GLOBAL
#define GLOBAL

#include <vector>

namespace nn::global {
using ValueType = float;
using ParamMetrix = std::vector<ValueType>;

struct Prediction {
	const int index;
	const global::ValueType value;
	Prediction() : index(0), value(0) {}
	Prediction(const int index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
};
} // namespace nn::global

#endif // GLOBAL
