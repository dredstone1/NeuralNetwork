#include "CNNetwork.hpp"
#include "tensor.hpp"
#include <vector>

namespace nn::model::cnn {
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(_config.getInputShape()),
      filters({3, 3, 1, 2}),
      activationMapN(makeActivationMapShape()),
      activationMapO(makeActivationMapShape()),
      output({nn::global::computeTensorSize(_config.getInputShape())}),
      activationFunction(_config.activation),
      visual(visual_) {
}

std::vector<size_t> CNNetwork::makeActivationMapShape() {
	std::vector<size_t> newShape = config.getInputShape();
	newShape.push_back(filters.getShape()[filters.getShape().size() - 1]);
	return newShape;
}

void CNNetwork::forward(const global::Tensor &newInput) {
	input = newInput;
	nn::global::tensor_gpu::conv2d(
	    input.gpu_data,
	    filters.gpu_data,
	    activationMapN.gpu_data,
	    28, 28,
	    1, 3);

	activationFunction.activate(activationMapN, activationMapO);

    output = activationMapO;
}

void CNNetwork::backward(global::Tensor **) {
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0;
}

void CNNetwork::resetGradient() {
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

void CNNetwork::updateWeights(IOptimizer &) {
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
