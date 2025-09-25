#include "lost_function.hpp"
#include "tensor.hpp"
#include "tensor_gpu.hpp"
#include <algorithm> // for std::max, std::min
#include <cmath>     // for std::log, std::abs

namespace nn::model {

// -----------------------------------------------------------------------------
// Public interface
// -----------------------------------------------------------------------------

global::ValueType Lost::LostF(const size_t index, const global::Tensor &outP,
                              const global::Tensor &outE) {
	switch (lostType) {
	case LostsType::CCE:
		return CCE(index, outE); // one-hot safe
	case LostsType::BCE:
		return BCE(index, outE); // one-hot safe
	case LostsType::MSE:
		return MSE(outP, outE); // statistical
	case LostsType::MAE:
		return MAE(outP, outE); // statistical
	}
	return 0; // fallback
}

// -----------------------------------------------------------------------------
// Private loss implementations
// -----------------------------------------------------------------------------

// One-hot categorical cross entropy
global::ValueType Lost::CCE(const size_t index, const global::Tensor &outE) {
	return -std::log(std::max(outE.getValue(index), MIN_LOSS_VALUE));
}

// One-hot binary cross entropy
global::ValueType Lost::BCE(const size_t index, const global::Tensor &outE) {
	global::ValueType p = std::max(
	    std::min(outE.getValue(index), static_cast<global::ValueType>(1.0) - MIN_LOSS_VALUE),
	    MIN_LOSS_VALUE);
	return -(p * std::log(p) + (1.0 - p) * std::log(1.0 - p));
}

// Mean Squared Error for statistical outputs
global::ValueType Lost::MSE(const global::Tensor &outP, const global::Tensor &outE) {
	if (global::Tensor::getGpuState()) {
		return 0;
	} else {
		global::ValueType sum = 0;
		size_t n = outP.numElements();
		for (size_t i = 0; i < n; ++i) {
			global::ValueType diff = outP.getValue(i) - outE.getValue(i);
			sum += diff * diff;
		}
		return sum / static_cast<global::ValueType>(n); // mean
	}
}

// Mean Absolute Error for statistical outputs
global::ValueType Lost::MAE(const global::Tensor &outP, const global::Tensor &outE) {
	if (global::Tensor::getGpuState()) {
		return 0;
	} else {
		global::ValueType sum = 0;
		size_t n = outP.numElements();
		for (size_t i = 0; i < n; ++i) {
			sum += std::abs(outP.getValue(i) - outE.getValue(i));
		}
		return sum / static_cast<global::ValueType>(n); // mean
	}
}

} // namespace nn::model
