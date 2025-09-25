#ifndef LOSS
#define LOSS

#include "dataBase.hpp"
#include "tensor.hpp"
#include "tensor_gpu.hpp"

namespace nn::model {

/// Minimum loss value to prevent numerical underflow
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

/**
 * @brief Types of loss functions supported by the framework.
 */
enum class LossType {
	CCE, ///< Classification (softmax + log, one-hot)
	BCE, ///< Binary Cross-Entropy (one-hot)
	MSE, ///< Mean Squared Error
	MAE  ///< Mean Absolute Error
};

class Loss {
  private:
	LossType lossType;

	// ----------------------
	// Private loss functions
	// ----------------------

	static OutType getOTypeFromLType(LossType lt);
	OutType getOTypeFromOut(const global::Tensor *outP);

	static global::ValueType CCE(const size_t index, const global::Tensor &outE);
	static global::ValueType BCE(const size_t index, const global::Tensor &outE);
	static global::ValueType MSE(const global::Tensor *outP, const global::Tensor &outE);
	static global::ValueType MAE(const global::Tensor *outP, const global::Tensor &outE);

  public:
	// ----------------------
	// Constructors & Destructor
	// ----------------------

	Loss(const LossType type, const OutType outType);
	Loss(const Loss &other) : lossType(other.lossType) {}
	~Loss() = default;

	// ----------------------
	// Public interface
	// ----------------------

	/**
	 * @brief Compute loss for a single element.
	 *
	 * @param index Index of the element (for one-hot / classification losses)
	 * @param outP Optional predictions (used by MSE/MAE, ignored for CCE/BCE)
	 * @param outE Ground-truth tensor
	 * @return Loss value
	 */
	global::ValueType LossF(const size_t index, const global::Tensor *outP,
	                        const global::Tensor &outE);
};

} // namespace nn::model

#endif // LOSS
