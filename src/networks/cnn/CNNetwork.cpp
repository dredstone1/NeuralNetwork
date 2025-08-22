#include "CNNetwork.hpp"
#include <random>

namespace nn::model::cnn {

CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(config.getInputShape()),
      filters({config.filterSize, config.filterSize, config.filterCount}),
      filtersGradient({config.filterSize, config.filterSize, config.filterCount}),
      activationMapN(makeActivationMapShape()),
      activationMapO(makeActivationMapShape()),
      output(config.getInputShape()),
      inputDelta(config.getInputShape()),
      activationDelta(makeActivationMapShape()),
      activationFunction(config.activation),
      visual(visual_) {
	std::vector<global::ValueType> tempFilters = randomFilters();
	filters = tempFilters;
}

std::vector<global::ValueType> CNNetwork::randomFilters() const {
	std::vector<global::ValueType> filtersTemp(filters.numElements());

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0, std::sqrt(2.0f / (config.filterSize * config.filterSize)));

	for (size_t i = 0; i < filtersTemp.size(); ++i) {
		filtersTemp[i] = dist(gen);
	}

	return filtersTemp;
}

void CNNetwork::conv2d_cpu() {
	Size size = getFeatureMapSize();

	for (size_t f = 0; f < config.filterCount; ++f) {
		for (size_t x = 0; x < size.h; ++x) {
			for (size_t y = 0; y < size.w; ++y) {
				global::ValueType sum = 0.0f;

				for (size_t i = 0; i < config.filterSize; ++i) {
					for (size_t j = 0; j < config.filterSize; ++j) {
						global::ValueType value = input.getValue({x + i, y + j});
						value *= filters.getValue({i, j, f});
						sum += value;
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
	newShape.push_back(config.filterCount);
	return newShape;
}

Size CNNetwork::getFeatureMapSize() {
	return {input.getShape()[0] - config.filterSize + 1,
	        input.getShape()[1] - config.filterSize + 1};
}

void CNNetwork::forward(const global::Tensor &newInput) {
	input.setData(newInput);

	if (nn::global::Tensor::getGpuState()) {
		nn::global::tensor_gpu::conv2d(
		    input.gpu_data, filters.gpu_data, activationMapN.gpu_data,
		    input.getShape()[0], input.getShape()[1],
		    config.filterCount, config.filterSize);
	} else {
		conv2d_cpu();
	}

	activationFunction.activate(activationMapN, activationMapO);

	output = activationMapO;
}

void CNNetwork::backward(global::Tensor **outputDeltas) {
	resetGradient();

	if (nn::global::Tensor::getGpuState()) {
		if (outputDeltas && *outputDeltas) {
			activationFunction.derivativeActivate(activationMapN, activationDelta);

			global::Tensor outputDelta = **outputDeltas;
			nn::global::tensor_gpu::multiply_vec(
			    activationDelta.gpu_data, outputDelta.gpu_data, activationDelta.gpu_data,
			    activationDelta.numElements());

			Size featureMapSize = getFeatureMapSize();
			nn::global::tensor_gpu::conv2d_backward_filter(
			    input.gpu_data, activationDelta.gpu_data, filtersGradient.gpu_data,
			    input.getShape()[0], input.getShape()[1], config.filterCount, config.filterSize,
			    featureMapSize.h, featureMapSize.w);

			nn::global::tensor_gpu::conv2d_backward_data(
			    activationDelta.gpu_data, filters.gpu_data, inputDelta.gpu_data,
			    featureMapSize.h, featureMapSize.w, config.filterCount, config.filterSize,
			    input.getShape()[0], input.getShape()[1]);
		}
	} else {
		if (outputDeltas && *outputDeltas) {
			activationFunction.derivativeActivate(activationMapN, activationDelta);

			global::Tensor outputDelta = **outputDeltas;
			for (size_t i = 0; i < activationDelta.numElements(); ++i) {
				activationDelta.setValue(i, activationDelta.getValue(i) * outputDelta.getValue(i));
			}

			calculateFilterGradients();

			calculateInputDelta(activationDelta);
		}
	}
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0.0f;
}

void CNNetwork::resetGradient() {
	filtersGradient.fill(0.0);
}

size_t CNNetwork::outputSize() const {
	return output.numElements();
}

const global::Tensor &CNNetwork::getOutput() const {
	return output;
}

const global::Tensor &CNNetwork::getInput() const {
	return input;
}

void CNNetwork::updateWeights(IOptimizer &optimizer) {
	optimizer.step(filters, filtersGradient);
}

void CNNetwork::calculateFilterGradients() {
	Size size = getFeatureMapSize();

	for (size_t f = 0; f < config.filterCount; ++f) {
		for (size_t i = 0; i < config.filterSize; ++i) {
			for (size_t j = 0; j < config.filterSize; ++j) {
				global::ValueType gradient = 0.0f;

				for (size_t x = 0; x < size.h; ++x) {
					for (size_t y = 0; y < size.w; ++y) {
						global::ValueType inputValue = input.getValue({x + i, y + j});
						global::ValueType deltaValue = activationDelta.getValue({x, y, f});
						gradient += inputValue * deltaValue;
					}
				}

				filtersGradient.setValue({i, j, f}, gradient);
			}
		}
	}
}

void CNNetwork::calculateInputDelta(const global::Tensor &deltas) {
	Size size = getFeatureMapSize();

	inputDelta.fill(0.0f);

	for (size_t x = 0; x < input.getShape()[0]; ++x) {
		for (size_t y = 0; y < input.getShape()[1]; ++y) {
			global::ValueType delta = 0.0f;

			for (size_t f = 0; f < config.filterCount; ++f) {
				for (size_t i = 0; i < config.filterSize; ++i) {
					for (size_t j = 0; j < config.filterSize; ++j) {
						size_t fm_x = x - i;
						size_t fm_y = y - j;

						if (fm_x >= 0 && fm_x < size.h &&
						    fm_y >= 0 && fm_y < size.w) {
							global::ValueType filterValue = filters.getValue({i, j, f});
							global::ValueType deltaValue = deltas.getValue({fm_x, fm_y, f});
							delta += filterValue * deltaValue;
						}
					}
				}
			}

			inputDelta.setValue({x, y}, delta);
		}
	}
}

std::vector<global::ValueType> CNNetwork::getParams() const {
	std::vector<global::ValueType> params;
	params.reserve(filters.numElements());

	for (size_t i = 0; i < filters.numElements(); ++i) {
		params.push_back(filters.getValue(i));
	}

	return params;
}

void CNNetwork::setParams(const global::Tensor &params) {
	if (params.numElements() == filters.numElements()) {
		for (size_t i = 0; i < filters.numElements(); ++i) {
			filters.setValue(i, params.getValue(i));
		}
	}
}

size_t CNNetwork::getParamCount() const {
	return filters.numElements();
}

void CNNetwork::setTraining(const bool) {
}

} // namespace nn::model::cnn
