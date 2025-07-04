#include "FNNetwork.hpp"
#include "Globals.hpp"
#include <memory>

namespace nn::model {
FNNetwork::FNNetwork(
    const FNNConfig &_config,
    const bool randomInit,
    const std::shared_ptr<visualizer::FnnVisualier> visual_)
    : config(_config),
      input(_config.getInputSize(), 0.0),
      visual(visual_) {
	int prevSize_ = _config.getInputSize();
	size_t i = 0;

	for (; i < _config.layersConfig.size(); i++) {
		layers.push_back(std::make_unique<Hidden_Layer>(
		    _config.layersConfig[i].size,
		    prevSize_,
		    _config.layersConfig[i].activationType,
		    randomInit));
		visual->initLayer(i, layers[i]->getDots(), layers[i]->getParms(), layers[i]->getGrad());

		prevSize_ = _config.layersConfig[i].size;
	}

	layers.push_back(std::make_unique<Output_Layer>(_config, prevSize_, randomInit));
	visual->initLayer(i, layers[i]->getDots(), layers[i]->getParms(), layers[i]->getGrad());
}

void FNNetwork::forward(const global::ParamMetrix &newInput) {
	input = newInput;
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
		visual->setUpdate();
	}
}

void FNNetwork::backward(const global::ParamMetrix &outputDeltas) {
	global::ParamMetrix deltas = outputDeltas;

	resetGradient();

	layers.back()->backward(deltas, layers[layers.size() - 2]->getOut());

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		const global::ParamMetrix &prev = (i == 0) ? input : layers[i - 1]->getOut();
		layers[i]->backward(deltas, prev, &layers[i + 1]->getParms());
		visual->setUpdate();
	}

	calculateInputDelta(deltas);
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

const global::ParamMetrix &FNNetwork::getNet() const {
	return layers[layers.size() - 1]->getNet();
}

const global::ParamMetrix &FNNetwork::getInput() const {
	return input;
}

void FNNetwork::updateWeights(const global::ValueType learningRate) {
	for (auto &layer : layers) {
		layer->updateWeight(learningRate);
	}
}

void FNNetwork::calculateInputDelta(const global::ParamMetrix &deltas) {
	for (int i = 0; i < inputSize(); i++) {
		for (size_t j = 0; j < layers[0]->getSize(); j++) {
			input[i] += deltas[j] * layers[0]->getWeight(j, i);
		}
	}
}
} // namespace nn::model
