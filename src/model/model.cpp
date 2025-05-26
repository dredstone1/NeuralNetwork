#include "model.hpp"
#include "neuralNetwork.hpp"
#include "visualizer/VisualizerController.hpp"
#include "visualizer/state.hpp"
#include <cmath>
#include <iostream>
#include <vector>

model::model(Config &_config, const bool use_visual) : network(_config.config_data.network_config), visual(_config.config_data.visualizer_config), useVisual(use_visual) {
	if (use_visual)
		visual.start(network);

	for (int i = 0; i < getLayerCount(); i++) {
		visual.update(i + 1, network.layers[i]->getParms());
	}
}

void model::run_model(const vector<double> &input) {
	run_model(input, network);
}

void print_vector(const vector<double> &metrix) {
	for (auto &value : metrix) {
		cout << value << " ";
	}

	cout << endl
	     << endl;
}

void model::run_model(const vector<double> &input, neural_network &temp_network) {
	visual.setNewPhaseMode(NNmode::Forword);
	visual.updateDots(0, input, input);
	temp_network.layers[0]->forward(input);
	visual.updateDots(1, temp_network.layers[0]->getOut(), temp_network.layers[0]->getNet());

	for (int i = 1; i < temp_network.getLayerCount(); i++) {
		temp_network.layers[i]->forward(temp_network.layers[i - 1]->getOut());
		visual.updateDots(i + 1, temp_network.layers[i]->getOut(), temp_network.layers[i]->getNet());
	}
}

void model::reset() {
	for (auto &layer : network.layers) {
		layer->reset();
	}
}

const vector<double> &model::getOutput() const {
	return network.layers[getHiddenLayerCount()]->getOut();
}

void model::updateWeights(const gradient &gradients) {
	visual.setNewPhaseMode(NNmode::Backward);

	for (int i = getLayerCount() - 1; i >= 0; i--) {
		getLayer(i).add(gradients.gradients[i]);
		visual.update(i + 1, getLayer(i).getParms());
	}
}
