#include "optimizers.hpp"
#include "tensor.hpp"
#include "tensor_gpu.hpp"

namespace nn::model {

void ConstantOptimizer::step(global::Tensor &weight, global::Tensor &grad) {
	// Scale learning rate by batch size
	global::ValueType div = config.getLearningRate() / batchSize;

	if (global::Tensor::getGpuState()) {
		// Perform weight update on GPU
		global::tensor_gpu::constStep(
		    weight.getGpuDataP(), // pointer to weights on GPU
		    grad.getGpuDataP(),   // pointer to gradients on GPU
		    weight.numElements(), // number of elements to update
		    div                   // learning rate scaling
		);
	} else {
		// Perform weight update on CPU
		for (size_t i = 0; i < weight.numElements(); ++i) {
			// w = w - grad * (lr / batchSize)
			global::ValueType value = weight.getValue(i) - (grad.getValue(i) * div);
			weight.setValue(i, value);
		}
	}
}

} // namespace nn::model
