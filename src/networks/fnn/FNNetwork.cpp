#include "FNNetwork.hpp"
#include "tensor.hpp"

namespace nn::model::fnn {
FNNetwork::FNNetwork(
    const FNNConfig &_config,
    const bool randomInit,
    const std::shared_ptr<visualizer::fnn::FnnVisualier> visual_)
    : config(_config),
      input({_config.getInputSize()}),
      visual(visual_) {
	int prevSize_ = _config.getInputSize();
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
	printf("test1: %zu", input.numElements());
	layers[0]->forward(input);
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

void FNNetwork::backward(const global::Tensor &outputDeltas) {
	global::Tensor deltas = outputDeltas;

	resetGradient();

	layers.back()->backward(deltas, layers[layers.size() - 2]->getOut());

	if (visual) {
		visual->setGrad(layers.size() - 1, layers[layers.size() - 1]->getGrad());
	}

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		const global::Tensor &prev = (i == 0) ? input : layers[i - 1]->getOut();
		layers[i]->backward(deltas, prev, &layers[i + 1]->getParms());

		if (visual) {
			visual->setGrad(i, layers[i]->getGrad());
		}

		vUpdate();
	}

	calculateInputDelta(deltas);
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

size_t FNNetwork::inputSize() const {
	return config.getInputSize();
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

void FNNetwork::calculateInputDelta(const global::Tensor &deltas) {
	input.fill(0);

	for (size_t i = 0; i < inputSize(); ++i) {
		for (size_t j = 0; j < layers[0]->size(); ++j) {
			input[i] += deltas[j] * layers[0]->getParms().weights({j, i});
		}
	}
}

size_t FNNetwork::getParamCount() const {
	size_t count = 0;

	for (auto &layer : layers) {
		count += layer->getParamCount();
	}

	return count;
}

global::Tensor FNNetwork::getParams() const {
	global::Tensor matrix({getParamCount()});

	size_t matrixI = 0;

	for (size_t i = 0; i < layers.size(); ++i) {
		global::Tensor params = layers[i]->getData();

		for (size_t j = 0; j < params.numElements(); ++j) {
			matrix[matrixI] = params[j];
			++matrixI;
		}
	}

	return matrix;
}

void FNNetwork::setParams(const global::Tensor params) {
	size_t j = 0;
	for (size_t i = 0; i < layers.size(); ++i) {
		global::Tensor newParam({layers[i]->getParamCount()});

		for (size_t k = 0; k < newParam.numElements(); ++k) {
			newParam[k] = params[j];
			++j;
		}

		layers[i]->setData(newParam);

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
