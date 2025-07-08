#ifndef GLOBAL
#define GLOBAL

#include <vector>

namespace nn::global {
using ValueType = float;
using ParamMetrix = std::vector<ValueType>;

struct FinalPrediction {
	const int index;
	const global::ValueType value;
	FinalPrediction() : index(0), value(0) {}
	FinalPrediction(const int index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
};

using Predictions = ParamMetrix;
} // namespace nn::global

#endif // GLOBAL
