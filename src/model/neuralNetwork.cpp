#include "neuralNetwork.hpp"

namespace nn::model {
NeuralNetwork::NeuralNetwork(const NetworkConfig &network_config)
    : config(network_config) {
	layers.reserve(network_config.hidden_layer_count() + 1);

	int prev_size = network_config.input_size;
	for (size_t i = 0; i < network_config.hidden_layer_count(); i++) {
		layers.emplace_back(std::make_unique<Hidden_Layer>(
		    network_config.layers_config[i].size,
		    prev_size,
		    network_config.layers_config[i].AT,
		    network_config.layers_config[i].weights_init_value));

		prev_size = network_config.layers_config[i].size;
	}

	layers.emplace_back(std::make_unique<Output_Layer>(
	    network_config.output_size,
	    prev_size,
	    network_config.output_init_value));
}

void NeuralNetwork::reset() {
	for (auto &layer : layers) {
		layer->reset();
	}
}
} // namespace nn
