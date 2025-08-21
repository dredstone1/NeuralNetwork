#include "CNNetwork.hpp"
#include "tensor.hpp"
#include <random>
#include <vector>

namespace nn::model::cnn {
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(config.getInputShape()),
      filters({config.filterSize, config.filterSize, 1, config.filterCount}),
      filtersGradient({config.filterSize, config.filterSize, 1, config.filterCount}),
      activationMapN(makeActivationMapShape()),
      activationMapO(makeActivationMapShape()),
      output({nn::global::computeTensorSize(_config.getInputShape())}),
      activationFunction(_config.activation),
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
	size_t outH = input.getShape()[0] - config.filterSize + 1;
	size_t outW = input.getShape()[1] - config.filterSize + 1;

	for (size_t f = 0; f < config.filterCount; ++f) {
		for (size_t x = 0; x < outH; ++x) {
			for (size_t y = 0; y < outW; ++y) {
				global::ValueType sum = 0.0f;

				for (size_t i = 0; i < config.filterSize; ++i) {
					for (size_t j = 0; j < config.filterSize; ++j) {
						sum += input.cpu_data[(x + i) * input.getShape()[1] + (y + j)] *
						       filters.cpu_data[f * config.filterSize * config.filterSize + i * config.filterSize + j];
					}
				}

				activationMapN.cpu_data[(f * outH + x) * outW + y] = sum;
			}
		}
	}
}

std::vector<size_t> CNNetwork::makeActivationMapShape() {
	std::vector<size_t> newShape = config.getInputShape();
	newShape.push_back(config.filterCount);
	return newShape;
}

void CNNetwork::forward(const global::Tensor &newInput) {
	input = newInput;

	if (nn::global::Tensor::getGpuState()) {
		nn::global::tensor_gpu::conv2d(
		    input.gpu_data,
		    filters.gpu_data,
		    activationMapN.gpu_data,
		    input.getShape()[0], input.getShape()[1],
		    config.filterCount, config.filterSize);
	} else {
		conv2d_cpu();
	}

	activationFunction.activate(activationMapN, activationMapO);

	output = activationMapO;
}

void CNNetwork::backward(global::Tensor **outputDeltas) {
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0;
}

void CNNetwork::resetGradient() {
	filtersGradient.fill(0.0);
}

size_t CNNetwork::outputSize() const {
	return config.getOutputSize();
}

const global::Tensor &CNNetwork::getOutput() const {
	return output;
}

const global::Tensor &CNNetwork::getInput() const {
	return input;
}

void CNNetwork::updateWeights(IOptimizer &optimizer) {
	// Use the provided optimizer to update the filters using the calculated gradients.
	optimizer.step(filters, filtersGradient);
}

void CNNetwork::calculateInputDelta(const global::Tensor &) {
}

std::vector<global::ValueType> CNNetwork::getParams() const {
	return std::vector<global::ValueType>();
}

void CNNetwork::setParams(const global::Tensor &) {
}

size_t CNNetwork::getParamCount() const {
	return 0;
}

void CNNetwork::setTraining(const bool) {
}
} // namespace nn::model::cnn
