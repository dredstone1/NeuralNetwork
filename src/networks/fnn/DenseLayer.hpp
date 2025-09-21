#ifndef DENSELAYER
#define DENSELAYER

#include "../src/model/optimizers.hpp"
#include <Globals.hpp>

namespace nn::model::fnn {

// ============================================================================
// CONSTANTS
// ============================================================================

/// Minimum loss value to prevent numerical underflow
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

/**
 * @struct LayerParams
 * @brief Container for layer parameters (weights and biases)
 * 
 * This structure holds the learnable parameters of a neural network layer,
 * including the weight matrix and bias vector. It provides convenient access
 * to layer dimensions and parameter counts.
 * 
 * @var global::Tensor weights The weight matrix (output_size × input_size)
 * @var global::Tensor biases The bias vector (output_size)
 * @var size_t size_ The output dimension of the layer
 * @var size_t prevSize_ The input dimension of the layer
 */
struct LayerParams {
	global::Tensor weights;  ///< Weight matrix for linear transformation
	global::Tensor biases;   ///< Bias vector for linear transformation

	size_t size_;      ///< Output dimension of the layer
	size_t prevSize_;  ///< Input dimension of the layer

	/**
	 * @brief Constructs layer parameters with specified dimensions
	 * @param out_dim Output dimension of the layer
	 * @param in_dim Input dimension of the layer
	 */
	LayerParams(size_t out_dim, size_t in_dim)
	    : weights({out_dim, in_dim}), biases({out_dim}),
	      size_(out_dim), prevSize_(in_dim) {}

	/**
	 * @brief Gets the output dimension of the layer
	 * @return The number of neurons in this layer
	 */
	size_t size() const { return size_; }
	
	/**
	 * @brief Gets the input dimension of the layer
	 * @return The number of inputs to this layer
	 */
	size_t prevSize() const { return prevSize_; }

	/**
	 * @brief Gets the total number of parameters in this layer
	 * @return The sum of weights and biases elements
	 */
	size_t paramSize() const { return biases.numElements() + weights.numElements(); }
};

/**
 * @class DenseLayer
 * @brief Base class for dense (fully connected) neural network layers
 * 
 * This is the base class for all dense layers in the neural network. It provides
 * the common functionality for forward and backward propagation, parameter management,
 * and gradient computation. Specific layer types (hidden, output) inherit from this
 * class and implement their specialized behavior.
 * 
 * @section features Key Features
 * - **Linear Transformation**: Matrix multiplication with weights and biases
 * - **Activation Functions**: Support for various activation functions
 * - **Gradient Computation**: Automatic gradient calculation for backpropagation
 * - **Parameter Management**: Easy access to weights, biases, and gradients
 * - **Training Modes**: Support for both training and inference modes
 * 
 * @section usage Usage Example
 * ```cpp
 * // Create a hidden layer with 128 neurons
 * Hidden_Layer layer(config, 784, true); // 784 inputs, random initialization
 * 
 * // Forward pass
 * layer.forward(input_tensor);
 * 
 * // Backward pass
 * layer.backward(&deltas, prev_layer_output, weight, &next_layer_params);
 * ```
 */
class DenseLayer {
  protected:
	global::Tensor net;           ///< Pre-activation values (z = Wx + b)
	global::Tensor out;           ///< Post-activation values (a = σ(z))

	LayerParams parameters;       ///< Layer parameters (weights and biases)
	LayerParams gradients;        ///< Parameter gradients for optimization

	global::Tensor deltaL;        ///< Layer delta for backpropagation

	Activation activationFunction; ///< Activation function for this layer

	bool isTraining{false};       ///< Training mode flag

	/**
	 * @brief Randomly initializes layer parameters
	 * 
	 * This method initializes the weights and biases with random values
	 * using appropriate initialization schemes (e.g., Xavier/He initialization).
	 */
	void fillParamRandom();

  public:
	// ========================================================================
	// CONSTRUCTORS AND DESTRUCTOR
	// ========================================================================

	/**
	 * @brief Constructs a dense layer with specified parameters
	 * @param size The number of neurons in this layer
	 * @param prevSize The number of inputs to this layer
	 * @param activation The activation function type to use
	 * @param randomInit Whether to randomly initialize parameters
	 */
	DenseLayer(
	    const size_t size,
	    const size_t prevSize,
	    const ActivationType activation,
	    const bool randomInit = false);
	
	/**
	 * @brief Virtual destructor for proper inheritance
	 */
	virtual ~DenseLayer() = default;

	// ========================================================================
	// PURE VIRTUAL METHODS (must be implemented by derived classes)
	// ========================================================================

	/**
	 * @brief Performs forward propagation through the layer
	 * @param metrix Input tensor from the previous layer
	 * 
	 * @note This is a pure virtual method that must be implemented by derived classes
	 * @note The implementation should compute: output = activation(weights * input + biases)
	 */
	virtual void forward(const global::Tensor &metrix) = 0;
	
	/**
	 * @brief Performs backward propagation through the layer
	 * @param deltas Pointer to delta tensor for gradient computation
	 * @param prevLayer Output from the previous layer
	 * @param weight Weight factor for gradient scaling
	 * @param nextLayer Parameters of the next layer (for hidden layers)
	 * 
	 * @note This is a pure virtual method that must be implemented by derived classes
	 * @note The implementation should compute gradients and propagate deltas backward
	 */
	virtual void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const global::ValueType weight,
	    const LayerParams *nextLayer = nullptr) = 0;

	// ========================================================================
	// COMMON METHODS
	// ========================================================================

	/**
	 * @brief Updates layer parameters using the specified optimizer
	 * @param optimizer The optimizer to use for parameter updates
	 * 
	 * @note This method applies the computed gradients to update the parameters
	 * @note The optimizer determines the specific update rule (SGD, Adam, etc.)
	 */
	void updateWeight(IOptimizer &optimizer);
	
	/**
	 * @brief Computes the loss for this layer (default implementation returns 0)
	 * @param prediction The target prediction for loss computation
	 * @return The computed loss value
	 * 
	 * @note This is typically only implemented by output layers
	 * @note Hidden layers usually return 0 as they don't compute loss directly
	 */
	virtual global::ValueType getLoss(const size_t , const global::Tensor &) { return 0; };

	// ========================================================================
	// ACCESSOR METHODS
	// ========================================================================

	/**
	 * @brief Gets the layer parameters (weights and biases)
	 * @return Const reference to the parameters structure
	 */
	const LayerParams &getParms() { return parameters; }
	
	/**
	 * @brief Gets the layer gradients
	 * @return Const reference to the gradients structure
	 */
	const LayerParams &getGrad() { return gradients; }

	/**
	 * @brief Gets the number of neurons in this layer
	 * @return The size of the layer
	 */
	size_t size() const { return net.numElements(); }
	
	/**
	 * @brief Gets the number of inputs to this layer
	 * @return The input dimension
	 */
	size_t prevSize() const { return parameters.prevSize(); }

	/**
	 * @brief Gets the pre-activation values
	 * @return Const reference to the net tensor
	 */
	const global::Tensor &getNet() const { return net; }
	
	/**
	 * @brief Gets the post-activation values
	 * @return Const reference to the output tensor
	 */
	const global::Tensor &getOut() const { return out; }

	// ========================================================================
	// UTILITY METHODS
	// ========================================================================

	/**
	 * @brief Resets the layer deltas to zero
	 * 
	 * @note This is called at the beginning of each training iteration
	 */
	void resetDots();
	
	/**
	 * @brief Resets the layer gradients to zero
	 * 
	 * @note This is called at the beginning of each training iteration
	 */
	void resetGradient();

	/**
	 * @brief Gets the total number of parameters in this layer
	 * @return The total parameter count
	 */
	size_t getParamCount() const;

	/**
	 * @brief Gets all layer parameters as a vector
	 * @return Vector containing all parameters (weights followed by biases)
	 */
	const std::vector<global::ValueType> getData() const;
	
	/**
	 * @brief Sets layer parameters from a tensor
	 * @param newParam The tensor containing new parameters
	 * @param offset The offset in the tensor to start reading from
	 * 
	 * @note This is used for loading pre-trained parameters
	 */
	void setData(const global::Tensor &newParam, const size_t offset);

	/**
	 * @brief Sets the training mode for this layer
	 * @param state True for training mode, false for inference mode
	 * 
	 * @note Training mode affects dropout and other regularization techniques
	 */
	void setTraining(const bool state) { isTraining = state; }
};

/**
 * @class Hidden_Layer
 * @brief Hidden layer implementation with dropout support
 * 
 * This class represents a hidden layer in a neural network. It extends the
 * base DenseLayer with additional features like dropout regularization for
 * preventing overfitting during training.
 * 
 * @section features Key Features
 * - **Dropout Regularization**: Randomly sets neurons to zero during training
 * - **Standard Forward/Backward Pass**: Implements the standard neural network operations
 * - **Gradient Computation**: Computes gradients for backpropagation
 * - **Training/Inference Modes**: Different behavior for training vs inference
 * 
 * @section dropout Dropout Regularization
 * During training, dropout randomly sets a fraction of neurons to zero to prevent
 * overfitting. During inference, all neurons are used but their outputs are scaled
 * by the dropout probability to maintain the expected output magnitude.
 */
class Hidden_Layer : public DenseLayer {
  private:
	const DenseLayerConfig &config;  ///< Configuration for this layer
	
	/**
	 * @brief Calculates the delta for this layer during backpropagation
	 * @param output The output from the next layer
	 * @param nextLayer The parameters of the next layer
	 * 
	 * @note This method computes the error signal to propagate backward
	 */
	void calculateDelta(
	    const global::Tensor &output,
	    const LayerParams &nextLayer);

	global::Tensor dropoutMask;  ///< Mask for dropout regularization

	/**
	 * @brief Creates a random dropout mask
	 * 
	 * @note This is called during training to randomly select neurons to drop
	 */
	void CreateDropoutMask();

  public:
	/**
	 * @brief Constructs a hidden layer with the specified configuration
	 * @param _config The configuration for this layer
	 * @param _prev_size The number of inputs to this layer
	 * @param randomInit Whether to randomly initialize parameters
	 */
	Hidden_Layer(
	    const DenseLayerConfig &_config,
	    const int _prev_size,
	    const bool randomInit = false)
	    : DenseLayer(_config.size, _prev_size, _config.activationType, randomInit),
	      config(_config),
	      dropoutMask({_config.size}) {}
	
	/**
	 * @brief Destructor
	 */
	~Hidden_Layer() override = default;

	/**
	 * @brief Performs forward propagation through the hidden layer
	 * @param metrix Input tensor from the previous layer
	 * 
	 * @note This computes: output = activation(weights * input + biases)
	 * @note During training, dropout is applied after activation
	 */
	void forward(const global::Tensor &metrix) override;
	
	/**
	 * @brief Performs backward propagation through the hidden layer
	 * @param deltas Pointer to delta tensor for gradient computation
	 * @param prevLayer Output from the previous layer
	 * @param weight Weight factor for gradient scaling
	 * @param nextLayer Parameters of the next layer
	 * 
	 * @note This computes gradients and propagates deltas backward
	 * @note The dropout mask is applied to the gradients during training
	 */
	void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const global::ValueType weight,
	    const LayerParams *nextLayer) override;
};

/**
 * @class Output_Layer
 * @brief Output layer implementation with loss computation
 * 
 * This class represents the output layer of a neural network. It extends the
 * base DenseLayer with loss computation capabilities for training. The output
 * layer is responsible for computing the final predictions and calculating
 * the loss for backpropagation.
 * 
 * @section features Key Features
 * - **Loss Computation**: Calculates cross-entropy loss for classification
 * - **Final Predictions**: Produces the final network output
 * - **Gradient Computation**: Computes gradients for the output layer
 * - **Softmax Support**: Optional softmax activation for multi-class classification
 * 
 * @section loss Cross-Entropy Loss
 * The output layer uses cross-entropy loss for classification tasks, which
 * measures the difference between predicted and actual class probabilities.
 */
class Output_Layer : public DenseLayer {
  private:
	const FNNConfig &config;  ///< Configuration for this layer

	/**
	 * @brief Computes cross-entropy loss for a single prediction
	 * @param prediction The predicted probabilities
	 * @param target The target class index
	 * @return The cross-entropy loss value
	 * 
	 * @note This is used for computing the loss during training
	 */
	static global::ValueType getCrossEntropyLoss(
	    const global::Tensor &prediction,
	    const size_t target);

  public:
	/**
	 * @brief Constructs an output layer with the specified configuration
	 * @param _config The configuration for this layer
	 * @param _prev_size The number of inputs to this layer
	 * @param randomInit Whether to randomly initialize parameters
	 */
	Output_Layer(
	    const FNNConfig &_config,
	    const int _prev_size,
	    const bool randomInit = false)
	    : DenseLayer(
	          _config.getOutputSize(),
	          _prev_size,
	          _config.outputActivation,
	          randomInit),
	      config(_config) {}
	
	/**
	 * @brief Destructor
	 */
	~Output_Layer() override = default;

	/**
	 * @brief Performs forward propagation through the output layer
	 * @param metrix Input tensor from the previous layer
	 * 
	 * @note This computes: output = activation(weights * input + biases)
	 * @note The activation is typically softmax for classification
	 */
	void forward(const global::Tensor &metrix) override;
	
	/**
	 * @brief Performs backward propagation through the output layer
	 * @param deltas Pointer to delta tensor for gradient computation
	 * @param prevLayer Output from the previous layer
	 * @param weight Weight factor for gradient scaling
	 * @param nextLayer Unused for output layer (always nullptr)
	 * 
	 * @note This computes the initial delta for backpropagation
	 * @note The delta is computed as the difference between prediction and target
	 */
	void backward(
	    global::Tensor **deltas,
	    const global::Tensor &prevLayer,
	    const global::ValueType weight,
	    const LayerParams *) override;

	/**
	 * @brief Computes the loss for this output layer
	 * @param index The target prediction for loss computation
	 * @return The computed cross-entropy loss
	 * 
	 * @note This is the main loss computation method for training
	 */
	global::ValueType getLoss(const size_t index, const global::Tensor &out) override;
};
} // namespace nn::model::fnn

#endif // DENSELAYER
