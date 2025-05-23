#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "Layers/layer.hpp"

typedef struct neural_network {
	vector<Layer *> layers;
	int const input_size;
	int const output_size;
	int const hidden_layers_size;
	int const hidden_layers_count;
	int getLayerCount() const { return (this->hidden_layers_count + 1); }
	neural_network(int _input_size, int _output_size, int _hidden_layers_size, int _hidden_layers_count);
	neural_network(const neural_network &other);
	void reset();
	~neural_network() = default;
} neural_network;

#endif // NEURAL_NETWORK
