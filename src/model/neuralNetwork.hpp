#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "config.hpp"
#include "layer.hpp"

namespace nn::model {
struct NeuralNetwork {
	std::vector<std::unique_ptr<ILayer>> layers;
	const NetworkConfig &config;

	NeuralNetwork(const NetworkConfig &networkConfig);
	~NeuralNetwork() = default;

	size_t getLayerCount() const { return layers.size(); }
	void reset();
};
} // namespace nn::model

#endif // NEURAL_NETWORK
