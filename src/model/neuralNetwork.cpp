#include "neuralNetwork.hpp"
#include "Layers/Hidden_Layer.hpp"
#include "Layers/Output_Layer.hpp"
#include "Layers/layer.hpp"
#include "config.hpp"
#include <cstddef>

neural_network::neural_network(const NetworkConfig &network_config)
    : config(network_config) {
	layers.reserve(network_config.hidden_layer_count() + 1);

	int prev_size = network_config.input_size;
	for (int i = 0; i < network_config.hidden_layer_count(); i++) {
		layers.emplace_back(new Hidden_Layer(
		    network_config.layers_config[i].size,
		    prev_size,
		    network_config.layers_config[i].AT,
		    network_config.layers_config[i].weights_init_value));

		prev_size = network_config.layers_config[i].size;
	}

	layers.emplace_back(new Output_Layer(
	    network_config.output_size,
	    prev_size,
	    network_config.output_init_value));
}

neural_network::neural_network(const neural_network &other)
    : config(other.config) {
	layers.reserve(other.getLayerCount() + 1);

	for (int i = 0; i < other.getLayerCount(); i++) {
		layers.emplace_back(other.layers[i]);
	}

	layers.emplace_back(other.layers[other.getLayerCount()]);
}

void neural_network::reset() {
	for (auto &layer : layers) {
		layer->reset();
	}
}

neural_network::~neural_network() {
	for (size_t i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}
