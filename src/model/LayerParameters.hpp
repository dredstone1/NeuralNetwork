#ifndef LAYER_PARAMETERS
#define LAYER_PARAMETERS

#include <Globals.hpp>
#include <cmath>

namespace nn::model {
constexpr global::ValueType PARAM_RESET_VALUE = 0;
constexpr int RN_ROUND_VALUE = 10000;

struct LayerParameters {
	std::vector<global::ParamMetrix> weights;
	global::ParamMetrix bias;

	LayerParameters(const int size, const int prevSize, const global::ValueType initValue);
	LayerParameters(const LayerParameters &other) : weights(other.weights) {}
	~LayerParameters() = default;

	void initializeParamRandom(const int prevSize);

	size_t getSize() const { return weights.size(); }
	size_t getPrevSize() const { return (weights.empty()) ? 0 : weights[0].size(); }

	void add(const LayerParameters &newGradient);
	void multiply(const global::ValueType value);
	void set(const LayerParameters &newGradient);
	void reset();
};
} // namespace nn::model
#endif // LAYER_PARAMETERS
