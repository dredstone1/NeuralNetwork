#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "Layers/layer.hpp"
#include "config.hpp"

typedef struct neural_network {
	vector<Layer *> layers;
	NetworkConfig &config;
	int getLayerCount() const { return (config.hidden_layer_count() + 1); }
	neural_network(NetworkConfig &network_config);
	neural_network(const neural_network &other);
	void reset();
	~neural_network() = default;
} neural_network;

#endif // NEURAL_NETWORK
