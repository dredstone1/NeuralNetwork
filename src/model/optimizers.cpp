#include "optimizers.hpp"

namespace nn::model {
void ConstantOptimizer::step(global::Tensor &weight, global::Tensor &grad) {
	grad *= config.getLearningRate() / batchSize;
	weight -= grad;
}
} // namespace nn::model
