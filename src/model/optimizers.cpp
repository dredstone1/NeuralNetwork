#include "optimizers.hpp"

namespace nn::model {

/**
 * @brief Performs a single optimization step using constant learning rate
 *
 * Updates the weight tensor by subtracting the scaled gradient. The gradient
 * is scaled by the learning rate divided by batch size to ensure proper
 * averaging across the batch.
 *
 * @param weight Weight tensor to be updated (modified in-place)
 * @param grad Gradient tensor containing computed gradients
 *
 * @note The gradient tensor is modified in-place during computation
 * @note Learning rate is divided by batch size for proper gradient averaging
 */
void ConstantOptimizer::step(global::Tensor &weight, global::Tensor &grad) {
	// Scale gradient by learning rate and batch size
	grad *= config.getLearningRate() / batchSize;

	// Update weights using gradient descent: w = w - α∇w
	weight -= grad;
}

} // namespace nn::model
