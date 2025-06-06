#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "Layers/layer.hpp"
#include "config.hpp"
#include <memory>

namespace nn {
struct neural_network {
	std::vector<std::unique_ptr<Layer>> layers;
	const NetworkConfig &config;
	int getLayerCount() const { return (config.hidden_layer_count() + 1); }
	neural_network(const NetworkConfig &network_config);
	void reset();
	~neural_network() = default;
};
} // namespace nn
#endif // NEURAL_NETWORK
