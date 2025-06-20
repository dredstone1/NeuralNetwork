#include "model.hpp"
#include "Globals.hpp"
#include <cmath>
#include <cstddef>

namespace nn::model {
Model::Model(Config &_config)
    : visual(_config.config_data) {
	visual.start();
}

void Model::runModel(const global::ParamMetrix &input) {
	network[0]->forward(input);

	for (size_t i = 1; i < network.size(); i++) {
		network[i]->forward(network[i - 1]->getOutput());
	}
}

// void Model::runModel(const global::ParamMetrix &input, const int modelIndex) {
// 	auto &subNetwork = network[modelIndex];
// 	visual.setNewPhaseMode(visualizer::NnMode::Forword);
//
// 	visual.updateDots(0, {input, input});
// 	temp_network.layers[0]->forward(input);
// 	visual.updateDots(1, {temp_network.layers[0]->getOut(), temp_network.layers[0]->getNet()});
//
// 	for (size_t i = 1; i < temp_network.getLayerCount(); i++) {
// 		temp_network.layers[i]->forward(temp_network.layers[i - 1]->getOut());
// 		visual.updateDots(i + 1, {temp_network.layers[i]->getOut(), temp_network.layers[i]->getNet()});
// 	}
// }

void Model::reset() {
	for (auto &subNetwork : network) {
		subNetwork.reset();
	}
}

const global::ParamMetrix &Model::getOutput() const {
	return network[network.size() - 1]->getOutput();
}

int Model::outputSize() {
	return network[network.size() - 1]->outputSize();
}

void Model::updateWeights(const global::ValueType learningRate) {
	visual.setNewPhaseMode(visualizer::NnMode::Backward);

	for (int i = network.size() - 1; i >= 0; i--) {
		network[i]->updateWeights(learningRate);
	}

	visual.setNewPhaseMode(visualizer::NnMode::Forword);

	// visual.update(gradients);
	//
	// for (int i = network.config.hidden_layer_count(); i >= 0; i--) {
	// 	getLayer(i).addParams(gradients.gradients[i]);
	// 	visual.update(i + 1, getLayer(i).getParms());
	// }
	//
}
} // namespace nn::model
