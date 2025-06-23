#include "FNNetwork.hpp"

namespace nn::model {
FNNetwork::FNNetwork(const FNNConfig &_config, const bool randomInit)
    : config(_config) {
	int prevSize = _config.inputSize;
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

void FNNetwork::forward(const global::ParamMetrix &input) {
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
	}
}

void FNNetwork::backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) {
	layers[layers.size() - 1]->backword(
	    output,
	    deltas,
	    layers[layers.size() - 2]->getOut(),
	    LayerParameters(0, 0));

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		global::ParamMetrix tempDeltas = deltas;

		if (i == 0) {
			layers[i]->backword(
			    tempDeltas,
			    deltas,
			    layers[i]->getNet(),
			    layers[i + 1]->getParms());
		} else {
			layers[i]->backword(
			    tempDeltas,
			    deltas,
			    layers[i - 1]->getOut(),
			    layers[i + 1]->getParms());
		}
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
