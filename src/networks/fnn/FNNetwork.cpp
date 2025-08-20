#include "FNNetwork.hpp"
#include "tensor.hpp"
#include <vector>

namespace nn::model::fnn {
FNNetwork::FNNetwork(
    const FNNConfig &_config,
    const bool randomInit,
    const std::shared_ptr<visualizer::fnn::FnnVisualier> visual_)
    : config(_config),
      input({_config.getInputShape()}),
      visual(visual_) {
	size_t prevSize_ = nn::global::computeTensorSize(_config.getInputShape());
	size_t i = 0;

	for (; i < _config.layersConfig.size(); ++i) {
		layers.push_back(std::make_unique<Hidden_Layer>(
		    _config.layersConfig[i],
		    prevSize_, randomInit));

		prevSize_ = _config.layersConfig[i].size;

		sendNewVData(i);
	}

	layers.push_back(std::make_unique<Output_Layer>(
	    _config,
	    prevSize_,
	    randomInit));

	sendNewVData(i);
}

void FNNetwork::sendNewVData(const size_t i) const {
	if (!visual) {
		return;
	}

	visual->setNet(i, layers[i]->getNet());
	visual->setOut(i, layers[i]->getOut());
	visual->setParam(i, layers[i]->getParms());
	visual->setGrad(i, layers[i]->getGrad());
}

void FNNetwork::sendNewVNeurons(const size_t i) const {
	if (!visual) {
		return;
	}

	visual->setNet(i, layers[i]->getNet());
	visual->setOut(i, layers[i]->getOut());
}

void FNNetwork::forward(const global::Tensor &newInput) {
	input = newInput;
	layers[0]->forward(newInput);
	sendNewVNeurons(0);

	for (size_t i = 1; i < layers.size(); ++i) {
		layers[i]->forward(layers[i - 1]->getOut());
		sendNewVNeurons(i);
	}

	vUpdate();
}

void FNNetwork::vUpdate() {
	if (!visual) {
		return;
	}

	visual->setUpdate();
	visual->attempPause();
}

void FNNetwork::backward(global::Tensor **outputDeltas) {
	resetGradient();

	layers.back()->backward(outputDeltas, layers[layers.size() - 2]->getOut());

	if (visual) {
		visual->setGrad(layers.size() - 1, layers[layers.size() - 1]->getGrad());
	}

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		const global::Tensor &prev = (i == 0) ? input : layers[i - 1]->getOut();
		layers[i]->backward(outputDeltas, prev, &layers[i + 1]->getParms());

		if (visual) {
			visual->setGrad(i, layers[i]->getGrad());
		}

		vUpdate();
	}

	calculateInputDelta(outputDeltas);
}

global::ValueType FNNetwork::getLoss(const global::Prediction &pre) const {
	return layers[layers.size() - 1]->getLoss(pre);
}

void FNNetwork::resetGradient() {
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->resetGradient();

		if (visual) {
			visual->setGrad(i, layers[i]->getGrad());
		}
	}
}

size_t FNNetwork::outputSize() const {
	return config.getOutputSize();
}

const global::Tensor &FNNetwork::getOutput() const {
	return layers[layers.size() - 1]->getOut();
}

const global::Tensor &FNNetwork::getInput() const {
	return input;
}

void FNNetwork::updateWeights(IOptimizer &optimizer) {
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->updateWeight(optimizer);

		if (visual) {
			visual->setParam(i, layers[i]->getParms());
		}
	}
}

void FNNetwork::calculateInputDelta(global::Tensor **deltas) {
	layers[0]->getParms().weights.matmulT(**deltas, input);
}

size_t FNNetwork::getParamCount() const {
	size_t count = 0;

	for (auto &layer : layers) {
		count += layer->getParamCount();
	}

	return count;
}

std::vector<global::ValueType> FNNetwork::getParams() const {
	std::vector<global::ValueType> matrix(getParamCount());

	size_t matrixI = 0;

	for (size_t i = 0; i < layers.size(); ++i) {
		std::vector<global::ValueType> params = layers[i]->getData();

		for (size_t j = 0; j < params.size(); ++j) {
			matrix[matrixI] = params[j];
			++matrixI;
		}
	}

	return matrix;
}

void FNNetwork::setParams(const global::Tensor &params) {
	size_t j = 0;
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->setData(params, j);
		j += layers[i]->getParamCount();

		if (visual) {
			visual->setParam(i, layers[i]->getParms());
		}
	}
}

void FNNetwork::setTraining(const bool state) {
	for (auto &layer : layers) {
		layer->setTraining(state);
	}
}
} // namespace nn::model::fnn
