#ifndef LAYER_PARAMETERS
#define LAYER_PARAMETERS

#include "Globals.hpp"
#include <cmath>
#include <vector>

namespace nn {
constexpr Global::ValueType PARAM_RESET_VALUE = 0;
constexpr int RN_ROUND_VALUE = 10000;

struct LayerParameters {
	LayerParameters(const int size, const int prev_size, const Global::ValueType init_value);
	LayerParameters(const LayerParameters &other);
	~LayerParameters() = default;
	std::vector<std::vector<Global::ValueType>> weights;
	size_t getSize() const { return weights.size(); }
	size_t getPrevSize() const { return (weights.empty()) ? 0 : weights[0].size(); }
	void add(const LayerParameters &new_gradient);
	void multiply(const double value);
	void set(const LayerParameters &new_gradient);
	void reset();
	void initialize_Param_rn(const int prev_size);
};
} // namespace nn
#endif // LAYER_PARAMETERS
