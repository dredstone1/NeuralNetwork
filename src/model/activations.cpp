#include "activations.hpp"
#include <stdexcept>

namespace nn::model {
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

void Activation::derivativeActivate(
    const nn::global::Tensor &net,
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

size_t Activation::getMaxElementIndex(const global::Tensor &metrix) {
	if (metrix.isGpu) {
		return global::tensor_gpu::getMaxElementIndex(metrix.gpu_data, metrix.gpu_data_size);
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

global::ValueType Activation::maxVector(const global::Tensor &metrix) {
	return metrix.getValue(getMaxElementIndex(metrix));
}

global::ValueType Activation::relu(const global::ValueType z) {
	return maxValue(z, 0.0f);
}
global::ValueType Activation::derivativeRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : 0.0;
}

global::ValueType Activation::leakyRelu(const global::ValueType z) {
	return (z > 0) ? z : RELU_LEAKY_ALPHA * z;
}
global::ValueType Activation::derivativeLeakyRelu(const global::ValueType z) {
	return (z > 0) ? 1.0 : RELU_LEAKY_ALPHA;
}

global::ValueType Activation::sigmoid(const global::ValueType z) {
	return 1.0 / (1.0 + std::exp(-z));
}
global::ValueType Activation::derivativeSigmoid(const global::ValueType z) {
	global::ValueType s = sigmoid(z);
	return s * (1.0 - s);
}

global::ValueType Activation::tanh(const global::ValueType z) {
	return std::tanh(z);
}
global::ValueType Activation::derivativeTanh(const global::ValueType z) {
	global::ValueType t = std::tanh(z);
	return 1.0 - t * t;
}

void Activation::relu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::relu(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = relu(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeRelu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::relu_derivative(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeRelu(net.cpu_data[i]);
		}
	}
}

void Activation::leakyRelu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::leaky_relu(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = leakyRelu(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeLeakyRelu(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::leaky_relu_derivative(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeLeakyRelu(net.cpu_data[i]);
		}
	}
}

void Activation::sigmoid(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::sigmoid(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = sigmoid(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeSigmoid(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::sigmoid_derivative(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeSigmoid(net.cpu_data[i]);
		}
	}
}

void Activation::tanh(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::tanh_activation(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] = tanh(net.cpu_data[i]);
		}
	}
}

void Activation::derivativeTanh(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::tanh_derivative(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		for (size_t i = 0; i < net.numElements(); ++i) {
			out.cpu_data[i] *= derivativeTanh(net.cpu_data[i]);
		}
	}
}

void Activation::softmax(const global::Tensor &net, global::Tensor &out) {
	if (net.isGpu) {
		global::tensor_gpu::softmax(net.gpu_data, out.gpu_data, net.gpu_data_size);
	} else {
		global::ValueType max = maxVector(net);
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
