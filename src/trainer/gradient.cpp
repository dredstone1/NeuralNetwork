#include "gradient.hpp"

namespace nn::training {
gradient::gradient(const model::NetworkConfig &_config) : config(_config) {
	gradients.reserve(config.hidden_layer_count() + 1);
	size_t i = 1;

	if (config.hidden_layer_count() > 0)
		gradients.emplace_back(
		    config.layers_config[0].size,
		    config.input_size,
		    _config.layers_config[0].weights_init_value);

	for (; i < config.hidden_layer_count(); i++) {
		gradients.emplace_back(
		    config.layers_config[i].size,
		    config.layers_config[i - 1].size,
		    _config.layers_config[i].weights_init_value);
	}

	gradients.emplace_back(
	    config.output_size,
	    (config.hidden_layer_count() > 0) ? config.layers_config[i - 1].size : config.input_size,
	    config.output_init_value);
}

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

void gradient::multiply(const global::ValueType value) {
	for (auto &gradient_layer : gradients) {
		gradient_layer.multiply(value);
	}
}
} // namespace nn::training
