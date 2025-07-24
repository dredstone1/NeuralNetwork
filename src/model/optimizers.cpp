#include "optimizers.hpp"

namespace nn::model {
void ConstantOptimizer::step(global::ValueType *weight, const global::ValueType *grad, std::size_t size) {
	for (std::size_t i = 0; i < size; ++i) {
		weight[i] -= config.getLearningRate() * grad[i] / static_cast<global::ValueType>(batchSize);
	}
}
} // namespace nn::model
