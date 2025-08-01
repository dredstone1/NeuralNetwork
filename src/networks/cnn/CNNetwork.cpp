#include "CNNetwork.hpp"

namespace nn::model::cnn {
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(_config.getInputSize(), 0.0),
      visual(visual_) {
}

void CNNetwork::forward(const global::ParamMetrix &newInput) {
	input = newInput;
}

void CNNetwork::backward(const global::ParamMetrix &) {
}

global::ValueType CNNetwork::getLoss(const global::Prediction &) const {
	return 0;
}

void CNNetwork::resetGradient() {
}

int CNNetwork::inputSize() const {
	return config.getInputSize();
}

int CNNetwork::outputSize() const {
	return config.getOutputSize();
}

const global::ParamMetrix &CNNetwork::getOutput() const {
	return input;
}

const global::ParamMetrix &CNNetwork::getInput() const {
	return input;
}

void CNNetwork::updateWeights(const std::shared_ptr<IOptimizer>) {
}

void CNNetwork::calculateInputDelta(const global::ParamMetrix &) {
}

global::ParamMetrix CNNetwork::getParams() const {
	return input;
}

void CNNetwork::setParams(const global::ParamMetrix) {
}

void CNNetwork::setTraining(const bool) {
}
} // namespace nn::model::cnn
