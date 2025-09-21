#include "CNNetwork.hpp"
#include "tensor.hpp"
#include <random>
#include <vector>

namespace nn::model::cnn {

/**
 * @brief Constructs a convolutional neural network layer
 * 
 * Creates a CNN layer with the specified configuration, including filter
 * dimensions, number of filters, and activation function. The layer can
 * be initialized with random weights or loaded from pre-trained parameters.
 * 
 * @param _config Configuration object containing CNN architecture
 * @param randomInit Whether to initialize filters randomly
 * @param visual_ Optional visualizer for real-time network monitoring
 */
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool randomInit,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(config.getInputShape()),
      filtersW(config.filterShape),
      filtersWGradient(config.filterShape),
      filtersB({config.filterShape[2]}),
      filtersBGradient({config.filterShape[2]}),
      activationMapN(makeActivationMapShape()),
      activationMapO(makeActivationMapShape()),
      activationDelta(makeActivationMapShape()),
      activationFunction(config.activation),
      visual(visual_) {
	if (randomInit) {
		initializeParameters();
	}
}

void CNNetwork::initializeParameters() {
	std::vector<global::ValueType> tempFilters = randomFilters();
	filtersW = tempFilters;
	filtersB.zero();
}

std::vector<global::ValueType> CNNetwork::randomFilters() const {
	std::vector<global::ValueType> filtersTemp(filtersW.numElements());
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<global::ValueType> dist(
	    0.0, std::sqrt(2.0f / (config.filterShape[0] * config.filterShape[1])));

	for (size_t i = 0; i < filtersTemp.size(); ++i) {
		filtersTemp[i] = dist(gen);
	}

	return filtersTemp;
}

/**
 * @brief Performs 2D convolution on CPU
 * 
 * Implements the 2D convolution operation using CPU computation. This method
 * applies multiple filters to the input feature map, computing the convolution
 * for each filter and storing the results in the activation map.
 * 
 * The convolution operation computes:
 * output[x,y,f] = bias[f] + sum(input[x+i, y+j, c] * filter[i, j, f, c])
 * 
 * @note This is the CPU implementation of the convolution operation
 * @note The operation is performed for all filters and output positions
 */
void CNNetwork::conv2d_cpu() {
	Size size = getFeatureMapSize();

	size_t filterCount = config.filterShape[2];
	size_t filterW = config.filterShape[0];
	size_t filterH = config.filterShape[1];
	size_t filterChannel = config.filterShape[3];

	for (size_t f = 0; f < filterCount; ++f) {
		for (size_t x = 0; x < size.w; ++x) {
			for (size_t y = 0; y < size.h; ++y) {
				global::ValueType sum = filtersB.getValue(f);

				for (size_t c = 0; c < filterChannel; ++c) {
					for (size_t i = 0; i < filterW; ++i) {
						for (size_t j = 0; j < filterH; ++j) {
							sum += input.getValue({x + i, y + j, c}) *
							       filtersW.getValue({i, j, f, c});
						}
					}
				}

				activationMapN.setValue({x, y, f}, sum);
			}
		}
	}
}

std::vector<size_t> CNNetwork::makeActivationMapShape() {
	Size featureMapSize = getFeatureMapSize();
	return {featureMapSize.w, featureMapSize.h, config.filterShape[2]};
}

Size CNNetwork::getFeatureMapSize() {
	return {input.getShape()[0] - config.filterShape[0] + 1,
	        input.getShape()[1] - config.filterShape[1] + 1};
}

/**
 * @brief Performs forward propagation through the CNN layer
 * 
 * Applies 2D convolution to the input feature map, followed by the activation
 * function. The operation is optimized for both CPU and GPU execution.
 * 
 * @param newInput Input feature map tensor
 * 
 * @note The input tensor should have shape [height, width, channels]
 * @note The output will have shape [output_height, output_width, num_filters]
 */
void CNNetwork::forward(const global::Tensor &newInput) {
	input.setData(newInput);

	if (nn::global::Tensor::getGpuState()) {
		size_t W = input.getShape()[0];
		size_t H = input.getShape()[1];
		size_t C = input.getShape()[2];
		size_t F = config.filterShape[2];
		size_t K = config.filterShape[0];

		nn::global::tensor_gpu::conv2d_multi_channel(
		    input.gpu_data, filtersW.gpu_data, filtersB.gpu_data,
		    activationMapN.gpu_data, H, W, C, F, K);
	} else {
		conv2d_cpu();
	}

	activationFunction.activate(activationMapN, activationMapO);
}

/**
 * @brief Performs backward propagation through the CNN layer
 * 
 * Computes gradients for the filters and biases, and propagates the error
 * signal backward to the input layer. This is used during training to
 * update the network parameters.
 * 
 * @param outputDeltas Pointer to delta tensor from the next layer
 * @param weight Weight factor for gradient scaling
 * 
 * @note This method computes gradients for both filters and biases
 * @note The input delta is computed and stored for propagation to previous layers
 */
void CNNetwork::backward(global::Tensor **outputDeltas,
                         const global::ValueType weight) {
	if (!outputDeltas || !*outputDeltas) {
		return;
	}

	resetGradient();

	activationFunction.derivativeActivate(activationMapO, **outputDeltas);
	activationDelta.setData(**outputDeltas);

	if (nn::global::Tensor::getGpuState()) {
		Size featureMapSize = getFeatureMapSize();

		nn::global::tensor_gpu::conv2d_multi_channel_backward_filter(
		    input.gpu_data, activationDelta.gpu_data, filtersWGradient.gpu_data,
		    input.getShape()[0], input.getShape()[1], config.filterShape[2],
		    config.filterShape[0],
		    featureMapSize.h, featureMapSize.w, input.getShape()[2], weight);

		nn::global::tensor_gpu::conv2d_multi_channel_backward_data(
		    activationDelta.gpu_data, filtersW.gpu_data, filtersB.gpu_data,
		    input.gpu_data, featureMapSize.h, featureMapSize.w,
		    config.filterShape[2], config.filterShape[0],
		    input.getShape()[0], input.getShape()[1], input.getShape()[2]);

		nn::global::tensor_gpu::conv2d_multi_channel_backward_bias(
		    activationDelta.gpu_data, filtersBGradient.gpu_data,
		    featureMapSize.h, featureMapSize.w, config.filterShape[2], weight);
	} else {
		calculateFilterGradients(weight);
		calculateBiasGradients(weight);

		calculateInputDelta(activationDelta);
	}
}

size_t CNNetwork::outputSize() const {
	return activationMapO.numElements();
}

global::ValueType CNNetwork::getLoss(const size_t, const global::Tensor &) const {
	return 0.0f;
}

void CNNetwork::resetGradient() {
	filtersBGradient.zero();
	filtersWGradient.zero();
}

const global::Tensor &CNNetwork::getOutput() const {
	return activationMapO;
}

global::Tensor *CNNetwork::getInput() {
	return &input;
}

std::shared_ptr<visualizer::IVisualNetwork> CNNetwork::getVisual() {
	return visual;
}

void CNNetwork::updateWeights(IOptimizer &optimizer) {
	optimizer.step(filtersW, filtersWGradient);
	optimizer.step(filtersB, filtersBGradient);
}

void CNNetwork::calculateFilterGradients(const global::ValueType weight) {
	Size size = getFeatureMapSize();

	size_t filterCount = config.filterShape[2];
	size_t filterW = config.filterShape[0];
	size_t filterH = config.filterShape[1];
	size_t filterChannel = config.filterShape[3];

	for (size_t f = 0; f < filterCount; ++f) {
		for (size_t c = 0; c < filterChannel; ++c) {
			for (size_t i = 0; i < filterW; ++i) {
				for (size_t j = 0; j < filterH; ++j) {
					global::ValueType gradient = 0.0f;

					for (size_t x = 0; x < size.w; ++x) {
						for (size_t y = 0; y < size.h; ++y) {
							global::ValueType inputValue =
							    input.getValue({x + i, y + j, c});
							global::ValueType deltaValue =
							    activationDelta.getValue({x, y, f});
							gradient += inputValue * deltaValue * weight;
						}
					}

					filtersWGradient.setValue({i, j, f, c}, gradient);
				}
			}
		}
	}
}

void CNNetwork::calculateBiasGradients(const global::ValueType weight) {
	Size size = getFeatureMapSize();
	size_t filterCount = config.filterShape[2];

	for (size_t f = 0; f < filterCount; ++f) {
		global::ValueType biasGradient = 0.0f;

		for (size_t x = 0; x < size.w; ++x) {
			for (size_t y = 0; y < size.h; ++y) {
				biasGradient += activationDelta.getValue({x, y, f}) * weight;
			}
		}

		filtersBGradient.setValue(f, biasGradient);
	}
}

void CNNetwork::calculateInputDelta(const global::Tensor &deltas) {
	Size size = getFeatureMapSize();

	size_t filterCount = config.filterShape[2];
	size_t filterW = config.filterShape[0];
	size_t filterH = config.filterShape[1];
	size_t filterChannel = config.filterShape[3];

	for (size_t x = 0; x < input.getShape()[0]; ++x) {
		for (size_t y = 0; y < input.getShape()[1]; ++y) {
			for (size_t c = 0; c < filterChannel; ++c) {
				global::ValueType delta = 0.0f;

				for (size_t f = 0; f < filterCount; ++f) {
					for (size_t i = 0; i < filterW; ++i) {
						for (size_t j = 0; j < filterH; ++j) {
							if (x < i || y < j)
								continue;

							size_t fm_x = x - i;
							size_t fm_y = y - j;

							if (fm_x < size.h && fm_y < size.w) {
								global::ValueType filterValue =
								    filtersW.getValue({i, j, f, c});
								global::ValueType deltaValue =
								    deltas.getValue({fm_x, fm_y, f});
								delta += filterValue * deltaValue;
							}
						}
					}
				}

				input.setValue({x, y, c}, delta);
			}
		}
	}
}

std::vector<global::ValueType> CNNetwork::getParams() const {
	std::vector<global::ValueType> params;
	params.reserve(filtersW.numElements() + filtersB.numElements());

	std::vector<global::ValueType> weights(filtersW.numElements());
	filtersW.getData(weights);
	params.insert(params.end(), weights.begin(), weights.end());

	std::vector<global::ValueType> bias(filtersB.numElements());
	filtersB.getData(bias);
	params.insert(params.end(), bias.begin(), bias.end());

	return params;
}

void CNNetwork::setParams(const global::Tensor &params) {
	size_t totalParams = filtersW.numElements() + filtersB.numElements();

	if (params.numElements() != totalParams) {
		return;
	}

	// Copy into weights
	filtersW.insertRange(params, 0, 0, filtersW.numElements());

	// Copy into biases
	filtersB.insertRange(params, filtersW.numElements(), 0, filtersB.numElements());
}

size_t CNNetwork::getParamCount() const {
	return filtersW.numElements() + filtersB.numElements();
}

} // namespace nn::model::cnn
