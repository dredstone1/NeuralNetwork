#include "CNNetwork.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <vector>

namespace nn::model::cnn {
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input({_config.getInputSize()}, 0.0),
      visual(visual_) {
}

void CNNetwork::forward(const global::Tensor &newInput) {
	input = newInput;
}

void CNNetwork::backward(global::Tensor **) {
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0;
}

void CNNetwork::resetGradient() {
}

size_t CNNetwork::inputSize() const {
	return config.getInputSize();
}

size_t CNNetwork::outputSize() const {
	return config.getOutputSize();
}

const global::Tensor &CNNetwork::getOutput() const {
	return input;
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
