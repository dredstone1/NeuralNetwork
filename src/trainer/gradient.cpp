#include "gradient.hpp"

using namespace std;

void gradient::add(const gradient &new_gradient) {
	for (size_t i = 0; i < gradients.size(); i++) {
		gradients.at(i).add(new_gradient.gradients.at(i));
	}
}

void gradient::reset() {
	for (auto &gradient_layer : gradients) {
		gradient_layer.reset();
	}
}

gradient::gradient(NetworkConfig &_config) : config(_config) {
	gradients.reserve(config.hidden_layer_count() + 1);
	int i = 1;

	if (config.hidden_layer_count() > 0)
		gradients.emplace_back(LayerParameters(config.layers_config[0].size, config.input_size));

	for (; i < config.hidden_layer_count(); i++) {
		gradients.emplace_back(LayerParameters(config.layers_config[i].size, config.layers_config[i - 1].size));
	}

	gradients.emplace_back(LayerParameters(config.output_size, (config.hidden_layer_count() > 0) ? config.layers_config[i - 1].size : config.input_size));
}

void gradient::multiply(const double value) {
	for (auto &gradient_layer : gradients) {
		gradient_layer.multiply(value);
	}
}

gradient::gradient(const gradient &other) : config(other.config) {
	gradients = other.gradients;
}
