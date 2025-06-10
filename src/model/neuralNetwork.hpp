#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "config.hpp"
#include "layer.hpp"

namespace nn::model {
struct NeuralNetwork {
	std::vector<std::unique_ptr<Layer>> layers;
	const NetworkConfig &config;

	NeuralNetwork(const NetworkConfig &networkConfig);
	size_t getLayerCount() const { return (config.hidden_layer_count() + 1); }
	void reset();
	~NeuralNetwork() = default;
};
} // namespace nn::model
#endif // NEURAL_NETWORK
