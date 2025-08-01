#include "CNNetwork.hpp"

namespace nn::model::cnn {
CNNetwork::CNNetwork(
    const CNNConfig &_config,
    const bool randomInit,
    const std::shared_ptr<visualizer::cnn::CnnVisualier> visual_)
    : config(_config),
      input(_config.getInputSize(), 0.0),
      visual(visual_) {
}

void CNNetwork::forward(const global::ParamMetrix &newInput) {
}

void CNNetwork::backward(const global::ParamMetrix &outputDeltas) {
}

global::ValueType CNNetwork::getLoss(const global::Prediction &pre) const {
}

void CNNetwork::resetGradient() {
}

int CNNetwork::inputSize() const {
}

int CNNetwork::outputSize() const {
}

const global::ParamMetrix &CNNetwork::getOutput() const {
}

const global::ParamMetrix &CNNetwork::getNet() const {
}

const global::ParamMetrix &CNNetwork::getInput() const {
	return input;
}

void CNNetwork::updateWeights(const std::shared_ptr<IOptimizer> optimizer) {
}

void CNNetwork::calculateInputDelta(const global::ParamMetrix &deltas) {
}

global::ParamMetrix CNNetwork::getParams() const {
}

void CNNetwork::setParams(const global::ParamMetrix params) {
}

void CNNetwork::setTraining(const bool state) {
}
} // namespace nn::model::cnn
