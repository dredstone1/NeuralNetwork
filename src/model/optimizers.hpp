/**
 * @file optimizers.cpp
 * @brief Implementation of optimization algorithms for neural network training
 * 
 * This file implements various optimization algorithms used to update neural
 * network parameters during training. Currently supports constant learning
 * rate optimization, with infrastructure for adding more sophisticated
 * optimizers like Adam, RMSprop, and momentum-based methods.
 */

#ifndef OPTIMIZERS
#define OPTIMIZERS

#include "config.hpp"
#include "tensor.hpp"
#include <cmath>

namespace nn::model {

/**
 * @class IOptimizer
 * @brief Abstract base class for optimization algorithms
 * 
 * This interface defines the contract that all optimization algorithms
 * must implement. It provides methods for updating parameters and
 * resetting optimizer state.
 * 
 * @section usage Usage Example
 * ```cpp
 * std::unique_ptr<IOptimizer> optimizer = std::make_unique<ConstantOptimizer>(config);
 * optimizer->setOfset(batch_size);
 * optimizer->step(weights, gradients);
 * ```
 */
class IOptimizer {
  protected:
	int batchSize;  ///< Current batch size for optimization

  public:
	/**
	 * @brief Virtual destructor for proper inheritance
	 */
	virtual ~IOptimizer() = default;

	/**
	 * @brief Updates parameters using the computed gradients
	 * @param weight The parameter tensor to update
	 * @param grad The gradient tensor for the parameters
	 * 
	 * @note This method applies the specific optimization algorithm
	 * @note The implementation depends on the concrete optimizer type
	 */
	virtual void step(global::Tensor &weight, global::Tensor &grad) = 0;
	
	/**
	 * @brief Resets the optimizer state
	 * 
	 * @note This method clears any internal state (momentum, etc.)
	 * @note Should be called at the beginning of each training epoch
	 */
	virtual void reset() = 0;

	/**
	 * @brief Sets the batch size for the optimizer
	 * @param batchSize_ The batch size to use
	 * 
	 * @note Some optimizers use batch size for parameter updates
	 */
	void setOfset(const int batchSize_) { batchSize = batchSize_; }
};

/**
 * @class ConstantOptimizer
 * @brief Simple constant learning rate optimizer (SGD)
 * 
 * This optimizer implements stochastic gradient descent with a constant
 * learning rate. It updates parameters by subtracting the scaled gradients
 * from the current parameter values.
 * 
 * Update rule: weight = weight - learning_rate * gradient
 */
class ConstantOptimizer : public IOptimizer {
  private:
	const ConstantOptimizerConfig &config;  ///< Configuration containing learning rate

  public:
	/**
	 * @brief Constructs a constant optimizer with the specified configuration
	 * @param config_ Configuration object containing learning rate and other parameters
	 */
	ConstantOptimizer(const ConstantOptimizerConfig &config_)
	    : config(config_) {}

	/**
	 * @brief Updates parameters using constant learning rate
	 * @param weight The parameter tensor to update
	 * @param grad The gradient tensor for the parameters
	 * 
	 * @note Implements: weight = weight - learning_rate * gradient
	 */
	void step(global::Tensor &weight, global::Tensor &grad) override;

	/**
	 * @brief Resets the optimizer state (no-op for constant optimizer)
	 * 
	 * @note Constant optimizer has no internal state to reset
	 */
	void reset() override {}
};
} // namespace nn::model

#endif // OPTIMIZERS
