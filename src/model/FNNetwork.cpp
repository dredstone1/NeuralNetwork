#include "FNNetwork.hpp"

namespace nn::model {
FNNetwork::FNNetwork(const FNNConfig &_config) : config(_config) {
	int prevSize = _config.inputSize;
	for (size_t i = 0; i < _config.layersConfig.size(); i++) {
		layers.push_back(std::make_unique<Hidden_Layer>(
		    _config.layersConfig[i].size,
		    prevSize,
		    _config.layersConfig[i].activationType));

		prevSize = _config.layersConfig[i].size;
	}

	layers.push_back(std::make_unique<Output_Layer>(_config, prevSize));
}

void FNNetwork::forward(const global::ParamMetrix &input) {
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
	}
}

void FNNetwork::backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) {
	int lastIndex = static_cast<int>(layers.size()) - 1;

	const auto &prevOut = (layers.size() > 1)
	                          ? layers[lastIndex - 1]->getOut()
	                          : layers[lastIndex]->getNet();
	layers[lastIndex]->backword(output, deltas, prevOut, LayerParameters(0, 0));

	for (int i = lastIndex - 1; i >= 0; --i) {
		global::ParamMetrix tempDeltas = deltas;

		const auto &input = (i > 0)
		                        ? layers[i - 1]->getOut()
		                        : layers[i]->getNet();
		const auto &nextParams = layers[i + 1]->getParms();

		layers[i]->backword(tempDeltas, deltas, input, nextParams);
	}
}

global::ValueType FNNetwork::getLost(const global::ParamMetrix &output) const {
	return layers[layers.size() - 1]->getLost(output);
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
