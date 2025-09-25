#include "FNNetwork.hpp"

namespace nn::model::fnn {

/**
 * @brief Constructs a fully connected neural network
 * 
 * Creates a neural network with the specified configuration, including
 * hidden layers and an output layer. The network can be initialized with
 * random weights or loaded from pre-trained parameters.
 * 
 * @param _config Configuration object containing network architecture
 * @param randomInit Whether to initialize weights randomly
 * @param visual_ Optional visualizer for real-time network monitoring
 */
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
		    _config.layersConfig[i], prevSize_, randomInit));
		prevSize_ = _config.layersConfig[i].size;
		sendNewVData(i);
	}

	layers.push_back(std::make_unique<Output_Layer>(_config, prevSize_,
	                                                randomInit));

	sendNewVData(i);
}

/**
 * @brief Sends layer data to the visualizer
 * 
 * Updates the visualizer with the current state of a specific layer,
 * including network values, outputs, parameters, and gradients.
 * 
 * @param i Index of the layer to update in the visualizer
 */
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

/**
 * @brief Performs forward propagation through the network
 * 
 * Propagates the input through all layers of the network, computing
 * activations for each layer. Updates the visualizer with intermediate
 * results if visualization is enabled.
 * 
 * @param newInput Input tensor to propagate through the network
 */
void FNNetwork::forward(const global::Tensor &newInput) {
	input = newInput;
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

/**
 * @brief Performs backward propagation through the network
 * 
 * Computes gradients for all layers by propagating the error signal
 * backward through the network. Updates the visualizer with gradient
 * information if visualization is enabled.
 * 
 * @param outputDeltas Pointer to delta tensor for gradient computation
 * @param weight Weight factor for gradient scaling
 */
void FNNetwork::backward(global::Tensor **outputDeltas, const global::ValueType weight) {
	resetGradient();

	layers.back()
	    ->backward(outputDeltas, layers[layers.size() - 2]->getOut(), weight);

	if (visual)
		visual->setGrad(layers.size() - 1, layers[layers.size() - 1]->getGrad());

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		const global::Tensor &prev = (i == 0) ? input : layers[i - 1]->getOut();
		layers[i]->backward(outputDeltas, prev, weight, &layers[i + 1]->getParms());

		if (visual) {
			visual->setGrad(i, layers[i]->getGrad());
			vUpdate();
		}
	}

	calculateInputDelta(outputDeltas);
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

global::Tensor *FNNetwork::getInput() {
	return &input;
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
