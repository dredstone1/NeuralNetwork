#include "CNNetwork.hpp"
#include <random>

namespace nn::model::cnn {

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
      output(config.getInputShape()),
      inputDelta(config.getInputShape()),
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
	filtersB.fill(0.0f);
}

std::vector<global::ValueType> CNNetwork::randomFilters() const {
	std::vector<global::ValueType> filtersTemp(filtersW.numElements());

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0, std::sqrt(2.0f / (config.filterShape[0] * config.filterShape[1])));

	for (size_t i = 0; i < filtersTemp.size(); ++i) {
		filtersTemp[i] = dist(gen);
	}

	return filtersTemp;
}

void CNNetwork::conv2d_cpu() {
	Size size = getFeatureMapSize();

	size_t filterCount = config.filterShape[2];
	size_t filterW = config.filterShape[0];
	size_t filterH = config.filterShape[1];
	size_t filterChannel = config.filterShape[3];

	for (size_t f = 0; f < filterCount; ++f) {
		for (size_t x = 0; x < size.h; ++x) {
			for (size_t y = 0; y < size.w; ++y) {
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

	std::vector<size_t> newShape = {featureMapSize.w, featureMapSize.h};
	newShape.push_back(config.filterShape[2]);
	return newShape;
}

Size CNNetwork::getFeatureMapSize() {
	return {input.getShape()[0] - config.filterShape[0] + 1,
	        input.getShape()[1] - config.filterShape[1] + 1};
}

void CNNetwork::forward(const global::Tensor &newInput) {
	input.setData(newInput);

	if (nn::global::Tensor::getGpuState()) {
		size_t H = input.getShape()[0];
		size_t W = input.getShape()[1];
		size_t C = input.getShape()[2];
		size_t F = config.filterShape[2];
		size_t K = config.filterShape[0];

		nn::global::tensor_gpu::conv2d_multi_channel(
		    input.gpu_data, filtersW.gpu_data, filtersB.gpu_data, activationMapN.gpu_data,
		    H, W, C, F, K);
	} else {
		conv2d_cpu();
	}

	activationFunction.activate(activationMapN, activationMapO);

	output = activationMapO;
}

void CNNetwork::backward(global::Tensor **outputDeltas) {
	if (!outputDeltas || !*outputDeltas) {
		return;
	}

	resetGradient();

	activationFunction.derivativeActivate(activationMapN, activationDelta);

	if (nn::global::Tensor::getGpuState()) {
		nn::global::tensor_gpu::multiply_vec(
		    activationDelta.gpu_data, (**outputDeltas).gpu_data, activationDelta.gpu_data,
		    activationDelta.numElements());

		Size featureMapSize = getFeatureMapSize();

		nn::global::tensor_gpu::conv2d_multi_channel_backward_filter(
		    input.gpu_data, activationDelta.gpu_data, filtersWGradient.gpu_data,
		    input.getShape()[0], input.getShape()[1], config.filterShape[2], config.filterShape[0],
		    featureMapSize.h, featureMapSize.w, input.getShape()[2]);

		nn::global::tensor_gpu::conv2d_multi_channel_backward_data(
		    activationDelta.gpu_data, filtersW.gpu_data, filtersB.gpu_data, inputDelta.gpu_data,
		    featureMapSize.h, featureMapSize.w, config.filterShape[2], config.filterShape[0],
		    input.getShape()[0], input.getShape()[1], input.getShape()[2]);

		nn::global::tensor_gpu::conv2d_multi_channel_backward_bias(
		    activationDelta.gpu_data, filtersBGradient.gpu_data,
		    featureMapSize.h, featureMapSize.w, config.filterShape[2]);
	} else {
		for (size_t i = 0; i < activationDelta.numElements(); ++i) {
			activationDelta.setValue(i, activationDelta.getValue(i) * (**outputDeltas).getValue(i));
		}

		calculateFilterGradients();
		calculateBiasGradients();

		calculateInputDelta(activationDelta);
	}
}

size_t CNNetwork::outputSize() const {
	return output.numElements();
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0.0f;
}

void CNNetwork::resetGradient() {
	filtersWGradient.fill(0.0);
	filtersBGradient.fill(0.0);
}

const global::Tensor &CNNetwork::getOutput() const {
	return output;
}

const global::Tensor &CNNetwork::getInput() const {
	return input;
}

void CNNetwork::updateWeights(IOptimizer &optimizer) {
	optimizer.step(filtersW, filtersWGradient);
	optimizer.step(filtersB, filtersBGradient);
}

void CNNetwork::calculateFilterGradients() {
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

					for (size_t x = 0; x < size.h; ++x) {
						for (size_t y = 0; y < size.w; ++y) {
							global::ValueType inputValue = input.getValue({x + i, y + j, c});
							global::ValueType deltaValue = activationDelta.getValue({x, y, f});
							gradient += inputValue * deltaValue;
						}
					}

					filtersWGradient.setValue({i, j, f, c}, gradient);
				}
			}
		}
	}
}

void CNNetwork::calculateBiasGradients() {
	Size size = getFeatureMapSize();
	size_t filterCount = config.filterShape[2];

	for (size_t f = 0; f < filterCount; ++f) {
		global::ValueType biasGradient = 0.0f;

		for (size_t x = 0; x < size.h; ++x) {
			for (size_t y = 0; y < size.w; ++y) {
				biasGradient += activationDelta.getValue({x, y, f});
			}
		}

		filtersBGradient.setValue(f, biasGradient);
	}
}

void CNNetwork::calculateInputDelta(const global::Tensor &deltas) {
	Size size = getFeatureMapSize();

	inputDelta.fill(0.0f);

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
								global::ValueType filterValue = filtersW.getValue({i, j, f, c});
								global::ValueType deltaValue = deltas.getValue({fm_x, fm_y, f});
								delta += filterValue * deltaValue;
							}
						}
					}
				}

				inputDelta.setValue({x, y, c}, delta);
			}
		}
	}
}

std::vector<global::ValueType> CNNetwork::getParams() const {
	std::vector<global::ValueType> params;
	params.reserve(filtersW.numElements() + filtersB.numElements());

	for (size_t i = 0; i < filtersW.numElements(); ++i) {
		params.push_back(filtersW.getValue(i));
	}

	for (size_t i = 0; i < filtersB.numElements(); ++i) {
		params.push_back(filtersB.getValue(i));
	}

	return params;
}

void CNNetwork::setParams(const global::Tensor &params) {
	size_t totalParams = filtersW.numElements() + filtersB.numElements();

	if (params.numElements() != totalParams) {
		return;
	}

	for (size_t i = 0; i < filtersW.numElements(); ++i) {
		filtersW.setValue(i, params.getValue(i));
	}

	for (size_t i = 0; i < filtersB.numElements(); ++i) {
		filtersB.setValue(i, params.getValue(filtersW.numElements() + i));
	}
}

size_t CNNetwork::getParamCount() const {
	return filtersW.numElements() + filtersB.numElements();
}

} // namespace nn::model::cnn
