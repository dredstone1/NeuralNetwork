#include "optimizers.hpp"

namespace nn::model {
global::ValueType ConstantOptimizer::calculate(const global::ValueType, const global::ValueType grad) {
	return config.getLearningRate() * grad / batchSize;
}
} // namespace nn::model
