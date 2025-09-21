#ifndef INETWORK
#define INETWORK

#include "../../src/model/optimizers.hpp"
#include "IvisualNetwork.hpp"
#include "tensor.hpp"

namespace nn::model {

/**
 * @class INetwork
 * @brief Abstract interface representing a complete neural network.
 *
 * INetwork defines the core interface for a trainable network, including
 * forward and backward passes, weight updates, parameter management, and
 * optional visualization.
 */
class INetwork {
  public:
	/** @brief Virtual destructor for proper cleanup of derived classes. */
	virtual ~INetwork() = default;

	/**
	 * @brief Performs the forward pass of the network.
	 * @param input Input tensor to propagate through the network.
	 */
	virtual void forward(const global::Tensor &input) = 0;

	/**
	 * @brief Performs the backward pass, computing gradients.
	 * @param outputDeltas Pointer to the gradient tensors from the next layer(s).
	 * @param weight Scaling factor for weight updates (usually learning rate).
	 */
	virtual void backward(global::Tensor **outputDeltas,
	                      const global::ValueType weight) = 0;

	/**
	 * @brief Updates network weights using the provided optimizer.
	 * @param optimizer Reference to an optimizer instance (e.g., SGD, Adam).
	 */
	virtual void updateWeights(IOptimizer &optimizer) = 0;

	/** @brief Resets all gradient accumulations to zero. */
	virtual void resetGradient() = 0;

	/**
	 * @brief Returns the size of the network output.
	 * @return Number of elements in the output tensor.
	 */
	virtual size_t outputSize() const = 0;

	/**
	 * @brief Computes the loss for a given prediction.
	 * @param index Prediction or target index.
	 * @return Loss value as a scalar.
	 */
	virtual global::ValueType getLoss(const size_t index, const global::Tensor &out) const = 0;

	/**
	 * @brief Returns a constant reference to the network output tensor.
	 * @return Output tensor of the network.
	 */
	virtual const global::Tensor &getOutput() const = 0;

	/**
	 * @brief Returns a pointer to the input tensor of the network.
	 * @return Input tensor.
	 */
	virtual global::Tensor *getInput() = 0;

	/**
	 * @brief Returns a shared pointer to the visual network for rendering.
	 * @return Visual network interface (IVisualNetwork) for this network.
	 */
	virtual std::shared_ptr<visualizer::IVisualNetwork> getVisual() = 0;

	/**
	 * @brief Retrieves all trainable parameters of the network as a vector.
	 * @return Vector of parameter values.
	 */
	virtual std::vector<global::ValueType> getParams() const = 0;

	/**
	 * @brief Sets the network's trainable parameters from a tensor.
	 * @param params Tensor containing new parameter values.
	 */
	virtual void setParams(const global::Tensor &params) = 0;

	/**
	 * @brief Sets the network's mode to training or evaluation.
	 * @param state true for training mode, false for evaluation.
	 */
	virtual void setTraining(const bool state) = 0;

	/**
	 * @brief Returns the total number of trainable parameters.
	 * @return Number of parameters in the network.
	 */
	virtual size_t getParamCount() const = 0;
};

} // namespace nn::model

#endif // INETWORK
