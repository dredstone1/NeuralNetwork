#include "activations.hpp"
#include <stdexcept>

namespace nn::model {

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
void Activation::activate(const global::Tensor &net, global::Tensor &out) const {
	if (net.numElements() != out.numElements()) {
		throw std::invalid_argument(
		    "Activation::activate: tensor size mismatch.\n"
		    "  net shape: " +
		    nn::global::shapeToString(net.getShape()) + "\n"
		                                                "  out shape: " +
		    nn::global::shapeToString(out.getShape()) + "\n"
		                                                "  element counts: " +
		    std::to_string(net.numElements()) +
		    " vs " + std::to_string(out.numElements()));
	}

	switch (activationType) {
	case ActivationType::Relu:
		relu(net, out);
		break;
	case ActivationType::LeakyRelu:
		leakyRelu(net, out);
		break;
	case ActivationType::Sigmoid:
		sigmoid(net, out);
		break;
	case ActivationType::Tanh:
		tanh(net, out);
		break;
	case ActivationType::Softmax:
		softmax(net, out);
		break;
	default:
		throw std::runtime_error(
		    "Activation::activate: unknown activation type (" +
		    std::to_string(static_cast<int>(activationType)) + ").");
	}
}

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
void Activation::derivativeActivate(const nn::global::Tensor &net,
                                    nn::global::Tensor &out) const {
	if (net.numElements() != out.numElements()) {
		throw std::invalid_argument(
		    "Activation::derivativeActivate: tensor size mismatch.\n"
		    "  net shape: " +
		    nn::global::shapeToString(net.getShape()) + "\n"
		                                                "  out shape: " +
		    nn::global::shapeToString(out.getShape()) + "\n"
		                                                "  element counts: " +
		    std::to_string(net.numElements()) +
		    " vs " + std::to_string(out.numElements()));
	}

	switch (activationType) {
	case ActivationType::Relu:
		derivativeRelu(net, out);
		break;
	case ActivationType::LeakyRelu:
		derivativeLeakyRelu(net, out);
		break;
	case ActivationType::Sigmoid:
		derivativeSigmoid(net, out);
		break;
	case ActivationType::Tanh:
		derivativeTanh(net, out);
		break;
	default:
		throw std::runtime_error(
		    "Activation::derivativeActivate: unknown activation type (" +
		    std::to_string(static_cast<int>(activationType)) + ").");
	}
}

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
size_t Activation::getMaxElementIndex(const global::Tensor &metrix) {
	if (metrix.isGpu) {
		return global::tensor_gpu::getMaxElementIndex(metrix.gpu_data,
		                                              metrix.gpu_data_size);
	}

	global::ValueType max_val = metrix.cpu_data[0];
	size_t max_index = 0;
	for (size_t i = 1; i < metrix.numElements(); ++i) {
		if (metrix.cpu_data[i] > max_val) {
			max_val = metrix.cpu_data[i];
			max_index = i;
		}
	}

	return max_index;
}

/**
 * @brief Gets the maximum value in a tensor
 *
 * This utility function returns the maximum value in the given tensor.
 * It's used internally by softmax for numerical stability.
 *
 * @param metrix The input tensor to search
 * @return The maximum value in the tensor
 */
global::ValueType Activation::maxVector(const global::Tensor &metrix) {
	return metrix.getValue(getMaxElementIndex(metrix));
}

// ============================================================================
// SCALAR ACTIVATION FUNCTIONS (CPU implementations)
// ============================================================================

/**
 * @brief ReLU activation function: f(x) = max(0, x)
 * @param z Input value
 * @return max(0, z)
 */
global::ValueType Activation::relu(const global::ValueType z) {
	return maxValue(z, 0.0f);
}

/**
 * @brief ReLU derivative: f'(x) = 1 if x > 0, else 0
 * @param z Input value
 * @return 1.0 if z > 0, 0.0 otherwise
 */
global::ValueType Activation::derivativeRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : 0.0;
}

/**
 * @brief Leaky ReLU activation function: f(x) = x if x > 0, else αx
 * @param z Input value
 * @return z if z > 0, otherwise RELU_LEAKY_ALPHA * z
 */
global::ValueType Activation::leakyRelu(const global::ValueType z) {
	return (z > 0) ? z : RELU_LEAKY_ALPHA * z;
}

/**
 * @brief Leaky ReLU derivative: f'(x) = 1 if x > 0, else α
 * @param z Input value
 * @return 1.0 if z > 0, RELU_LEAKY_ALPHA otherwise
 */
global::ValueType Activation::derivativeLeakyRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

/**
 * @brief Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * @param z Input value
 * @return Sigmoid of z, bounded between 0 and 1
 */
global::ValueType Activation::sigmoid(const global::ValueType z) {
	return 1.0 / (1.0 + std::exp(-z));
}

/**
 * @brief Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
 * @param z Input value
 * @return Derivative of sigmoid at z
 */
global::ValueType Activation::derivativeSigmoid(const global::ValueType z) {
	const global::ValueType s = sigmoid(z);
	return s * (1.0 - s);
}

/**
 * @brief Tanh activation function: f(x) = tanh(x)
 * @param z Input value
 * @return Hyperbolic tangent of z, bounded between -1 and 1
 */
global::ValueType Activation::tanh(const global::ValueType z) {
	return std::tanh(z);
}

/**
 * @brief Tanh derivative: f'(x) = 1 - tanh²(x)
 * @param z Input value
 * @return Derivative of tanh at z
 */
global::ValueType Activation::derivativeTanh(const global::ValueType z) {
	const global::ValueType t = std::tanh(z);
	return 1.0 - t * t;
}

// ============================================================================
// VECTORIZED ACTIVATION FUNCTIONS (CPU/GPU implementations)
// ============================================================================

/**
 * @brief Vectorized ReLU activation
 *
 * Applies ReLU activation to all elements in the input tensor and stores
 * the result in the output tensor. Uses optimized implementations for
 * both CPU and GPU execution.
 *
 * @param net Input tensor containing pre-activation values
 * @param out Output tensor to store post-activation values
 */
void Activation::relu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::relu(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = relu(net.cpu_data[i]);
		}
	}
}

/**
 * @brief Vectorized ReLU derivative
 *
 * Applies ReLU derivative to all elements in the input tensor and multiplies
 * the result with the output tensor (in-place operation). Used during
 * backpropagation for gradient computation.
 *
 * @param net Input tensor containing pre-activation values
 * @param out Output tensor to be modified in-place with derivative values
 */
void Activation::derivativeRelu(const global::Tensor &net,
                                global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::relu_derivative(net.gpu_data, out.gpu_data,
		                                    net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeRelu(net.cpu_data[i]);
		}
	}
}

void Activation::leakyRelu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::leaky_relu(net.gpu_data, out.gpu_data,
		                               net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = leakyRelu(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeLeakyRelu(const global::Tensor &net,
                                     global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::leaky_relu_derivative(net.gpu_data, out.gpu_data,
		                                          net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeLeakyRelu(net.cpu_data[i]);
		}
	}
}

void Activation::sigmoid(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::sigmoid(net.gpu_data, out.gpu_data,
		                            net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = sigmoid(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeSigmoid(const global::Tensor &net,
                                   global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::sigmoid_derivative(net.gpu_data, out.gpu_data,
		                                       net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeSigmoid(net.cpu_data[i]);
		}
	}
}

void Activation::tanh(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::tanh_activation(net.gpu_data, out.gpu_data,
		                                    net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = tanh(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeTanh(const global::Tensor &net,
                                global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::tanh_derivative(net.gpu_data, out.gpu_data,
		                                    net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeTanh(net.cpu_data[i]);
		}
	}
}

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
void Activation::softmax(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::softmax(net.gpu_data, out.gpu_data,
		                            net.gpu_data_size);
	} else {
		const global::ValueType max = maxVector(net);
		global::ValueType sum = 0.0;

		for (size_t i = 0; i < net.numElements(); ++i) {
			global::ValueType x = net.cpu_data[i] - max;
			if (x < -700.0)
				x = -700.0;
			if (x > 700.0)
				x = 700.0;
			out.cpu_data[i] = std::exp(x);
			sum += out.cpu_data[i];
		}

		sum = maxValue(sum, 1e-10);

		out /= sum;
	}
}
} // namespace nn::model
