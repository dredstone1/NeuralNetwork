#ifndef ACTIVATIONS
#define ACTIVATIONS

#include "tensor.hpp"
#include <cmath>

namespace nn::model {

// ============================================================================
// ACTIVATION CONSTANTS
// ============================================================================

/// Leaky ReLU negative slope parameter (alpha value)
constexpr global::ValueType RELU_LEAKY_ALPHA = 0.01;

/**
 * @brief Utility function to find the maximum of two values
 * @param a First value (ValueType)
 * @param b Second value (float)
 * @return The maximum of the two values
 */
constexpr global::ValueType maxValue(const global::ValueType &a, const float &b) {
	return (a < b) ? b : a;
}

/**
 * @enum ActivationType
 * @brief Enumeration of available activation function types
 *
 * This enum defines all the activation functions supported by the neural
 * network library. Each type corresponds to a specific mathematical function
 * used to introduce non-linearity into neural networks.
 */
enum class ActivationType {
	Relu,      ///< Rectified Linear Unit: f(x) = max(0, x)
	LeakyRelu, ///< Leaky ReLU: f(x) = x if x > 0, else αx
	Sigmoid,   ///< Sigmoid: f(x) = 1 / (1 + e^(-x))
	Tanh,      ///< Hyperbolic Tangent: f(x) = tanh(x)
	Softmax,   ///< Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j))
	None,      ///< No activation (identity function)
};

/**
 * @class Activation
 * @brief Manages activation functions for neural network layers
 *
 * This class provides a unified interface for applying various activation
 * functions to tensors. It supports both forward propagation (activation)
 * and backward propagation (derivative) operations. All operations are
 * optimized for both CPU and GPU execution modes.
 *
 * @section features Supported Activation Functions
 * - **ReLU**: Rectified Linear Unit for addressing vanishing gradient problem
 * - **Leaky ReLU**: Modified ReLU that allows small negative values
 * - **Sigmoid**: S-shaped curve for binary classification
 * - **Tanh**: Hyperbolic tangent for bounded outputs
 * - **Softmax**: Normalized exponential for multi-class classification
 * - **None**: Identity function (no activation)
 *
 * @section usage Usage Example
 * ```cpp
 * Activation relu(ActivationType::Relu);
 * Tensor input({10}, 1.0f);
 * Tensor output({10});
 *
 * relu.activate(input, output);        // Forward pass
 * relu.derivativeActivate(input, output); // Backward pass
 * ```
 */
class Activation {
  private:
	const ActivationType activationType; ///< The type of activation function

	// ========================================================================
	// SCALAR ACTIVATION FUNCTIONS (CPU implementations)
	// ========================================================================

	/// ReLU activation: f(x) = max(0, x)
	static global::ValueType relu(const global::ValueType x);
	/// ReLU derivative: f'(x) = 1 if x > 0, else 0
	static global::ValueType derivativeRelu(const global::ValueType x);

	/// Leaky ReLU activation: f(x) = x if x > 0, else αx
	static global::ValueType leakyRelu(const global::ValueType x);
	/// Leaky ReLU derivative: f'(x) = 1 if x > 0, else α
	static global::ValueType derivativeLeakyRelu(const global::ValueType x);

	/// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
	static global::ValueType sigmoid(const global::ValueType z);
	/// Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
	static global::ValueType derivativeSigmoid(const global::ValueType z);

	/// Tanh activation: f(x) = tanh(x)
	static global::ValueType tanh(const global::ValueType z);
	/// Tanh derivative: f'(x) = 1 - tanh²(x)
	static global::ValueType derivativeTanh(const global::ValueType z);

	// ========================================================================
	// VECTORIZED ACTIVATION FUNCTIONS (CPU/GPU implementations)
	// ========================================================================

	/// Vectorized ReLU activation
	static void relu(const global::Tensor &net, global::Tensor &out);
	/// Vectorized ReLU derivative
	static void derivativeRelu(const global::Tensor &net, global::Tensor &out);

	/// Vectorized Leaky ReLU activation
	static void leakyRelu(const global::Tensor &net, global::Tensor &out);
	/// Vectorized Leaky ReLU derivative
	static void derivativeLeakyRelu(const global::Tensor &net,
	                                global::Tensor &out);

	/// Vectorized Sigmoid activation
	static void sigmoid(const global::Tensor &net, global::Tensor &out);
	/// Vectorized Sigmoid derivative
	static void derivativeSigmoid(const global::Tensor &net,
	                              global::Tensor &out);

	/// Vectorized Tanh activation
	static void tanh(const global::Tensor &net, global::Tensor &out);
	/// Vectorized Tanh derivative
	static void derivativeTanh(const global::Tensor &net, global::Tensor &out);

	/**
	 * @brief Vectorized Softmax activation
	 *
	 * Applies softmax activation to the input tensor, which normalizes the values
	 * to create a probability distribution. The softmax function is:
	 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
	 *
	 * This implementation includes numerical stability measures to prevent
	 * overflow by subtracting the maximum value before computing exponentials.
	 *
	 * @param net Input tensor containing pre-activation values
	 * @param out Output tensor to store normalized probabilities
	 *
	 * @note The output values sum to 1.0 (probability distribution)
	 * @note Uses numerical stability tricks to prevent overflow
	 */
	static void softmax(const global::Tensor &net, global::Tensor &out);

	/// Utility function to find maximum value in a tensor
	static global::ValueType maxVector(const global::Tensor &metrix);

  public:
	// ========================================================================
	// CONSTRUCTORS AND DESTRUCTOR
	// ========================================================================

	/**
	 * @brief Constructs an activation function of the specified type
	 * @param activationType_ The type of activation function to create
	 */
	Activation(const ActivationType activationType_)
	    : activationType(activationType_) {}

	/**
	 * @brief Copy constructor
	 * @param other The activation function to copy
	 */
	Activation(const Activation &other)
	    : activationType(other.activationType) {}

	/**
	 * @brief Destructor (default)
	 */
	~Activation() = default;

	// ========================================================================
	// PUBLIC INTERFACE
	// ========================================================================

	/**
	 * @brief Applies the activation function to a tensor
	 *
	 * This method applies the configured activation function to the input tensor
	 * and stores the result in the output tensor. The operation is performed
	 * element-wise across the entire tensor.
	 *
	 * @param net Input tensor containing pre-activation values
	 * @param out Output tensor to store post-activation values
	 *
	 * @throws std::invalid_argument If input and output tensors have different sizes
	 * @throws std::runtime_error If the activation type is unknown
	 *
	 * @note The output tensor must have the same shape as the input tensor
	 * @note This method automatically chooses between CPU and GPU implementations
	 */
	void activate(const global::Tensor &net, global::Tensor &out) const;

	/**
	 * @brief Applies the derivative of the activation function to a tensor
	 *
	 * This method computes the derivative of the configured activation function
	 * and applies it to the input tensor. This is used during backpropagation
	 * to compute gradients for the previous layer.
	 *
	 * @param net Input tensor containing pre-activation values
	 * @param out Output tensor to store derivative values (modified in-place)
	 *
	 * @throws std::invalid_argument If input and output tensors have different sizes
	 * @throws std::runtime_error If the activation type is unknown
	 *
	 * @note The output tensor is modified in-place (multiplied by the derivative)
	 * @note This method automatically chooses between CPU and GPU implementations
	 */
	void derivativeActivate(const global::Tensor &net,
	                        global::Tensor &out) const;

	/**
	 * @brief Gets the type of this activation function
	 * @return The ActivationType of this activation function
	 */
	ActivationType getType() { return activationType; }

	/**
	 * @brief Finds the index of the maximum element in a tensor
	 *
	 * This utility function finds the index of the element with the maximum value
	 * in the given tensor. It's commonly used with softmax activation for
	 * classification tasks.
	 *
	 * @param metrix The input tensor to search
	 * @return The index of the element with the maximum value
	 *
	 * @note This function works with both CPU and GPU tensors
	 * @note For GPU tensors, it uses optimized CUDA kernels
	 */
	static size_t getMaxElementIndex(const global::Tensor &metrix);
};
} // namespace nn::model

#endif // ACTIVATIONS
