/**
 * @file FNNetwork.hpp
 * @brief Header file for Fully Connected Neural Network implementation
 * 
 * This file defines the FNNetwork class, which implements a fully connected
 * (dense) neural network using multiple dense layers. It provides forward
 * and backward propagation, parameter management, and optional visualization
 * capabilities for monitoring network behavior during training.
 */

#ifndef FNNNETWORK
#define FNNNETWORK

#include "FnnVisualizer.hpp"
#include <network/INetwork.hpp>

namespace nn::model::fnn {

/**
 * @class FNNetwork
 * @brief Fully Connected Neural Network implementation
 * 
 * The FNNetwork class represents a feedforward neural network composed of
 * multiple dense (fully connected) layers. It implements the INetwork interface
 * to provide standardized methods for forward propagation, backpropagation,
 * weight updates, and parameter management.
 * 
 * Key features:
 * - Multiple dense layers with configurable architectures
 * - Forward and backward propagation with automatic gradient calculation
 * - Integration with various optimization algorithms
 * - Optional real-time visualization of network state
 * - Parameter serialization for model saving/loading
 * - Loss calculation for different prediction types
 * 
 * The network supports various activation functions and can be configured
 * through the FNNConfig object passed during construction.
 */
class FNNetwork : public INetwork {
  private:
	const FNNConfig &config;                                      ///< Configuration for network architecture
	std::vector<std::unique_ptr<DenseLayer>> layers;             ///< Collection of dense layers
	global::Tensor input;                                         ///< Input tensor storage

	const std::shared_ptr<visualizer::fnn::FnnVisualier> visual; ///< Optional visualizer component

	/** @brief Calculates input deltas for gradient propagation to previous layers */
	void calculateInputDelta(global::Tensor **deltas);

	/** @brief Updates visualization data if visualizer is enabled */
	void vUpdate();

	/** @brief Sends new data values to visualizer for specified layer */
	void sendNewVData(const size_t i) const;
	
	/** @brief Sends neuron activation values to visualizer for specified layer */
	void sendNewVNeurons(const size_t i) const;

  public:
	/**
	 * @brief Constructs a fully connected network with specified configuration
	 * 
	 * @param _config Network configuration containing layer sizes, activations, etc.
	 * @param randomInit Whether to initialize weights randomly (true) or as zeros (false)
	 * @param visual_ Optional visualizer for real-time network monitoring
	 */
	FNNetwork(
	    const FNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::fnn::FnnVisualier> visual_ = nullptr);
	
	/** @brief Default virtual destructor */
	~FNNetwork() override = default;

	/**
	 * @brief Performs forward propagation through the network
	 * @param newInput Input tensor to process through the network
	 */
	void forward(const global::Tensor &newInput) override;
	
	/**
	 * @brief Performs backward propagation to compute gradients
	 * @param outputDeltas Gradient tensors from subsequent layers
	 * @param weight Scaling factor for gradients (e.g., sample weight)
	 */
	void backward(global::Tensor **outputDeltas, const global::ValueType weight) override;
	
	/**
	 * @brief Updates network weights using the provided optimizer
	 * @param optimizer Optimization algorithm to use for weight updates
	 */
	void updateWeights(IOptimizer &optimizer) override;
	
	/** @brief Resets all accumulated gradients to zero */
	void resetGradient() override;

	/**
	 * @brief Calculates loss for a given prediction target
	 * @param index Target prediction information
	 * @return Computed loss value
	 */
	global::ValueType getLoss(const global::Prediction &index) const override;

	/**
	 * @brief Gets the size of the network's output layer
	 * @return Number of output neurons
	 */
	size_t outputSize() const override;

	/**
	 * @brief Gets the network's output tensor (read-only)
	 * @return Reference to the output tensor
	 */
	const global::Tensor &getOutput() const override;
	
	/**
	 * @brief Gets pointer to the network's input tensor for modification
	 * @return Pointer to the input tensor
	 */
	global::Tensor *getInput() override;

	/**
	 * @brief Gets the visualization component for this network
	 * @return Shared pointer to the visualizer, or nullptr if disabled
	 */
	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override {
		return visual;
	}

	/**
	 * @brief Extracts all network parameters as a flat vector
	 * @return Vector containing all weights and biases
	 */
	std::vector<global::ValueType> getParams() const override;
	
	/**
	 * @brief Sets network parameters from a flat tensor
	 * @param params Tensor containing all weights and biases to set
	 */
	void setParams(const global::Tensor &params) override;

	/**
	 * @brief Gets the total number of trainable parameters
	 * @return Total count of weights and biases in the network
	 */
	size_t getParamCount() const override;

	void setTraining(const bool state) override;
};
} // namespace nn::model::fnn

#endif // FNNNETWORK
