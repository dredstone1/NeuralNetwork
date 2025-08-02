#include "optimizers.hpp"

namespace nn::model {
void ConstantOptimizer::step(global::Tensor &weight, const global::Tensor &grad, size_t size) {
	for (std::size_t i = 0; i < size; ++i) {
		weight({i}) -= config.getLearningRate() * grad({i}) / batchSize;
	}
}
} // namespace nn::model
