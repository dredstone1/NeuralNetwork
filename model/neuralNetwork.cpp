#include "neuralNetwork.hpp"
#include "Layers/Hidden_Layer.hpp"
#include "Layers/Output_Layer.hpp"
#include "Layers/layer.hpp"

neural_network::neural_network(const int _input_size, const int _output_size, const int _hidden_layers_size, const int _hidden_layers_count) : input_size(_input_size), output_size(_output_size), hidden_layers_size(_hidden_layers_size), hidden_layers_count(_hidden_layers_count) {
	layers.reserve(_hidden_layers_count + 1);

	int prev_size = input_size;
	for (int i = 0; i < hidden_layers_count; i++) {
		Layer *temp = new Hidden_Layer(hidden_layers_size, prev_size);
		layers.push_back(temp);
		prev_size = hidden_layers_size;
	}

	layers.push_back(new Output_Layer(_output_size, prev_size));
}

neural_network::neural_network(neural_network const &other) : input_size(other.input_size), output_size(other.output_size), hidden_layers_size(other.hidden_layers_size), hidden_layers_count(other.hidden_layers_count) {
	layers.reserve(hidden_layers_count + 1);

	for (int i = 0; i < hidden_layers_count; i++) {
		layers.push_back(new Hidden_Layer(*(other.layers[i])));
	}

	layers.push_back(new Output_Layer(*(other.layers[hidden_layers_count])));
}

void neural_network::reset() {
	for (auto &layer : layers) {
		layer->reset();
	}
}
