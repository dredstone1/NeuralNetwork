#include "optimizers.hpp"

namespace nn::model {
void ConstantOptimizer::step(float *weight, const float *grad, std::size_t size) {
	const float lr = config.getLearningRate() / static_cast<float>(batchSize);

	for (std::size_t i = 0; i < size; ++i) {
		weight[i] -= lr * grad[i];
	}
}
} // namespace nn::model
