#include "neuralNetwork.hpp"
#include "Layers/Hidden_Layer.hpp"
#include "Layers/Output_Layer.hpp"
#include "Layers/layer.hpp"
#include "model/config.hpp"

neural_network::neural_network(NetworkConfig &network_config) : config(network_config) {
	layers.reserve(network_config.hidden_layer_count() + 1);

	int prev_size = network_config.input_size;
	for (int i = 0; i < network_config.hidden_layer_count(); i++) {
		Layer *temp = new Hidden_Layer(network_config.layers_config[i].size, prev_size, network_config.layers_config[i].activation);
		layers.push_back(temp);
		prev_size = network_config.layers_config[i].size;
	}

	layers.push_back(new Output_Layer(network_config.output_size, prev_size));
}

neural_network::neural_network(const neural_network &other) : config(other.config) {
	layers.reserve(other.getLayerCount() + 1);

	for (int i = 0; i < other.getLayerCount(); i++) {
		layers.push_back(new Hidden_Layer((Hidden_Layer &)*(other.layers[i])));
	}

	layers.push_back(new Output_Layer(*(other.layers[other.getLayerCount()])));
}

void neural_network::reset() {
	for (auto &layer : layers) {
		layer->reset();
	}
}
