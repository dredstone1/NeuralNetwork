#include "optimizers.hpp"
#include "tensor.hpp"
#include "tensor_gpu.hpp"

namespace nn::model {
void ConstantOptimizer::step(global::Tensor &weight, global::Tensor &grad) {
	global::ValueType div = config.getLearningRate() / batchSize;
	if (global::Tensor::getGpuState()) {
		global::tensor_gpu::constStep(weight.getGpuDataP(),
		                              grad.getGpuDataP(), weight.numElements(),
		                              div);
	} else {
		for (size_t i = 0; i < weight.numElements(); ++i) {
			global::ValueType value = weight.getValue(i) - (grad.getValue(i) * div);
			weight.setValue(i, value);
		}
	}
}
} // namespace nn::model
