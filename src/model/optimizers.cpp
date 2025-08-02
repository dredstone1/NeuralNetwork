#include "optimizers.hpp"

namespace nn::model {
void ConstantOptimizer::step(global::Tensor &weight, const global::Tensor &grad) {
	weight -= grad * (config.getLearningRate() / batchSize);
}
} // namespace nn::model
