/**
 * @file CNNetwork.hpp
 * @brief Header file for Convolutional Neural Network implementation
 * 
 * This file defines the CNNetwork class, which implements a convolutional neural
 * network for processing spatial data such as images. It provides 2D convolution
 * operations, activation functions, and gradient computation for training CNNs
 * with backpropagation.
 */

#ifndef CNNNETWORK
#define CNNNETWORK

#include "CnnVisualizer.hpp"
#include <network/INetwork.hpp>

namespace nn::model::cnn {

/**
 * @struct Size
 * @brief Simple structure to represent 2D dimensions
 * 
 * Used to specify spatial dimensions for feature maps, filters,
 * and other 2D data structures in the CNN.
 */
struct Size {
	size_t w; ///< Width dimension
	size_t h; ///< Height dimension
};

/**
 * @class CNNetwork
 * @brief Convolutional Neural Network implementation
 * 
 * The CNNetwork class implements a convolutional neural network capable of
 * processing 2D spatial data through convolution operations. It supports
 * multiple filter channels, bias terms, and various activation functions.
 * 
 * Key features:
 * - 2D convolution operations with configurable filter sizes
 * - Multiple feature maps with independent filters
 * - Bias terms for each filter channel
 * - Forward and backward propagation with gradient computation
 * - Integration with visualization system for real-time monitoring
 * - CPU-based convolution implementation with potential GPU acceleration
 * 
 * The network processes input data through convolution, applies activation
 * functions, and computes gradients for training via backpropagation.
 */
class CNNetwork : public INetwork {
  private:
	const CNNConfig &config;        ///< Configuration for CNN architecture
	global::Tensor input;           ///< Input tensor storage

	global::Tensor filtersW;        ///< Convolution filter weights
	global::Tensor filtersWGradient; ///< Gradients for filter weights

	global::Tensor filtersB;        ///< Bias terms for each filter
	global::Tensor filtersBGradient; ///< Gradients for bias terms

	global::Tensor activationMapN;  ///< Pre-activation feature maps (before activation function)
	global::Tensor activationMapO;  ///< Post-activation feature maps (after activation function)

	global::Tensor activationDelta; ///< Delta values for backpropagation

	Activation activationFunction;  ///< Activation function for feature maps

	/** @brief Calculates input gradients for backpropagation to previous layers */
	void calculateInputDelta(const global::Tensor &deltas);
	
	/** @brief Computes gradients for convolutional filter weights */
	void calculateFilterGradients(const global::ValueType weight);
	
	/** @brief Computes gradients for bias terms */
	void calculateBiasGradients(const global::ValueType weight);
	
	/** @brief Initializes filter weights and biases */
	void initializeParameters();

	const std::shared_ptr<visualizer::cnn::CnnVisualier> visual; ///< Optional visualizer component

	/** @brief Creates shape vector for activation map tensors */
	std::vector<size_t> makeActivationMapShape();

	/** @brief Generates random initial values for filter weights */
	std::vector<global::ValueType> randomFilters() const;

	/** @brief Performs 2D convolution operation using CPU implementation */
	void conv2d_cpu();

	/** @brief Calculates the size of output feature maps after convolution */
	Size getFeatureMapSize();

  public:
	/**
	 * @brief Constructs a convolutional neural network with specified configuration
	 * 
	 * @param _config CNN configuration containing filter sizes, stride, padding, etc.
	 * @param randomInit Whether to initialize weights randomly (true) or as zeros (false)
	 * @param visual_ Optional visualizer for real-time network monitoring
	 */
	CNNetwork(
	    const CNNConfig &_config,
	    const bool randomInit,
	    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_ =
	        std::shared_ptr<visualizer::cnn::CnnVisualier>());
	
	/** @brief Default virtual destructor */
	~CNNetwork() override = default;

	/**
	 * @brief Performs forward propagation through the convolutional layer
	 * 
	 * Applies 2D convolution operation to the input using learned filters,
	 * adds bias terms, and applies the activation function to produce
	 * feature maps.
	 * 
	 * @param newInput Input tensor with spatial dimensions to convolve
	 */
	void forward(const global::Tensor &newInput) override;
	
	/**
	 * @brief Performs backward propagation to compute gradients
	 * 
	 * Computes gradients for filters, biases, and input based on the
	 * error signals from subsequent layers. Updates accumulated gradients
	 * for later weight updates.
	 * 
	 * @param outputDeltas Gradient tensors from subsequent layers
	 * @param weight Scaling factor for gradients (e.g., sample weight)
	 */
	void backward(global::Tensor **outputDeltas,
	              const global::ValueType weight) override;
	
	/**
	 * @brief Updates network parameters using the provided optimizer
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
	 * @brief Gets the size of the network's output (number of feature maps)
	 * @return Number of output feature maps
	 */
	size_t outputSize() const override;

	/**
	 * @brief Gets the network's output tensor (read-only)
	 * @return Reference to the output activation maps
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
	std::shared_ptr<visualizer::IVisualNetwork> getVisual() override;

	/**
	 * @brief Extracts all network parameters as a flat vector
	 * @return Vector containing all filter weights and biases
	 */
	std::vector<global::ValueType> getParams() const override;
	
	/**
	 * @brief Sets network parameters from a flat tensor
	 * @param params Tensor containing all weights and biases to set
	 */
	void setParams(const global::Tensor &params) override;

	/**
	 * @brief Gets the total number of trainable parameters
	 * @return Total count of filter weights and biases in the network
	 */
	size_t getParamCount() const override;

	void setTraining(const bool) override {}
};

} // namespace nn::model::cnn

#endif // CNNNETWORK
