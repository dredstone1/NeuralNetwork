#include "FNNetwork.hpp"
#include "Globals.hpp"
#include "config.hpp"

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

	for (; i < _config.layersConfig.size(); ++i) {
		layers.push_back(std::make_unique<Hidden_Layer>(
		    _config.layersConfig[i],
		    prevSize_,
		    randomInit));
		visual->initLayer(
		    i,
		    layers[i]->getDots(),
		    layers[i]->getParms(),
		    layers[i]->getGrad());

		prevSize_ = _config.layersConfig[i].size;
	}

	layers.push_back(std::make_unique<Output_Layer>(
	    _config,
	    prevSize_,
	    randomInit));
	visual->initLayer(
	    i,
	    layers[i]->getDots(),
	    layers[i]->getParms(),
	    layers[i]->getGrad());
}

void FNNetwork::forward(const global::ParamMetrix &newInput) {
	input = newInput;
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); ++i) {
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

global::ValueType FNNetwork::getLoss(const global::ParamMetrix &pre) const {
	return layers[layers.size() - 1]->getLoss(pre);
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
	for (int i = 0; i < inputSize(); ++i) {
		for (size_t j = 0; j < layers[0]->getSize(); ++j) {
			input[i] += deltas[j] * layers[0]->getParms().weights[j][i];
		}
	}
}

global::ParamMetrix FNNetwork::getParams() const {
	global::ParamMetrix matrix;

	for (size_t i = 0; i < layers.size(); ++i) {
		global::ParamMetrix params = layers[i]->getData();

		matrix.insert(matrix.end(), params.begin(), params.end());
	}

	return matrix;
}

void FNNetwork::setParams(const global::ParamMetrix params) {
	global::ParamMetrix matrix;

	int j = 0;
	for (size_t i = 0; i < layers.size(); ++i) {
		global::ParamMetrix newParam(layers[i]->getSize() * layers[i]->getPrevSize() + layers[i]->getSize());

		for (size_t k = 0; k < newParam.size(); ++k) {
			newParam[k] = params[j];

			j++;
		}

		layers[i]->setData(newParam);
	}
}
} // namespace nn::model
