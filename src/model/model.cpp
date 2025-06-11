#include "model.hpp"
#include <cmath>

namespace nn::model {
Model::Model(Config &_config, const bool useVisual)
    : network(_config.config_data.network_config),
      visual(_config.config_data),
      useVisual(useVisual) {
	if (useVisual)
		visual.start(network);
}

void Model::runModel(const global::ParamMetrix &input) {
	runModel(input, network);
}

void Model::runModel(const global::ParamMetrix &input, NeuralNetwork &temp_network) {
	visual.setNewPhaseMode(visualizer::NnMode::Forword);

	visual.updateDots(0, {input, input});
	temp_network.layers[0]->forward(input);
	visual.updateDots(1, {temp_network.layers[0]->getOut(), temp_network.layers[0]->getNet()});

	for (size_t i = 1; i < temp_network.getLayerCount(); i++) {
		temp_network.layers[i]->forward(temp_network.layers[i - 1]->getOut());
		visual.updateDots(i + 1, {temp_network.layers[i]->getOut(), temp_network.layers[i]->getNet()});
	}
}

void Model::reset() {
	for (auto &layer : network.layers) {
		layer->reset();
	}
}

const global::ParamMetrix &Model::getOutput() const {
	return network.layers[getHiddenLayerCount()]->getOut();
}

void Model::updateWeights(const training::gradient &gradients) {
	visual.setNewPhaseMode(visualizer::NnMode::Backward);
	visual.update(gradients);

	for (int i = network.config.hidden_layer_count(); i >= 0; i--) {
		getLayer(i).addParams(gradients.gradients[i]);
		visual.update(i + 1, getLayer(i).getParms());
	}

	visual.setNewPhaseMode(visualizer::NnMode::Forword);
}
} // namespace nn::model
