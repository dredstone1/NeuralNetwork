#include "FNNetwork.hpp"

namespace nn::model {
FNNetwork::FNNetwork(const FNNConfig &_config, const bool randomInit)
    : config(_config),
      input(_config.getInputSize(), 0.0) {
	int prevSize = _config.getInputSize();
	for (size_t i = 0; i < _config.layersConfig.size(); i++) {
		layers.push_back(std::make_unique<Hidden_Layer>(
		    _config.layersConfig[i].size,
		    prevSize,
		    _config.layersConfig[i].activationType,
		    randomInit));

		prevSize = _config.layersConfig[i].size;
	}

	layers.push_back(std::make_unique<Output_Layer>(_config, prevSize, randomInit));
}

void FNNetwork::forward(const global::ParamMetrix &newInput) {
	input = newInput;
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
	}
}

void FNNetwork::backward(global::ParamMetrix &deltas) {
	resetGradient();

	layers[layers.size() - 1]->backward(deltas, layers[layers.size() - 2]->getOut());

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		auto &prevLayer = (i == 0) ? input : layers[i - 1]->getOut();

		layers[i]->backward(deltas, prevLayer, &layers[i + 1]->getParms());
	}
}

global::ValueType FNNetwork::getLoss(const int index) const {
	return layers[layers.size() - 1]->getLoss(index);
}

void FNNetwork::resetGradient() {
	for (auto &layer : layers) {
		layer->resetGradient();
	}
}

int FNNetwork::inputSize() const {
	return layers[0]->getPrevSize();
}

int FNNetwork::outputSize() const {
	return layers[layers.size() - 1]->getSize();
}

const global::ParamMetrix &FNNetwork::getOutput() const {
	return layers[layers.size() - 1]->getOut();
}

void FNNetwork::updateWeights(const global::ValueType learningRate) {
	for (auto &layer : layers) {
		layer->updateWeight(learningRate);
	}
}
} // namespace nn::model
