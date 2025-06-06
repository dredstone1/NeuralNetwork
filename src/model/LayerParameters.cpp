#include "LayerParameters.hpp"
#include <random>

namespace nn {
LayerParameters::LayerParameters(const int size, const int prev_size, const Global::ValueType init_value) {
	weights.resize(size, std::vector<Global::ValueType>(prev_size, init_value));

	if (init_value < 0.0)
		initialize_Param_rn(prev_size);
}

void LayerParameters::initialize_Param_rn(const int prev_size) {
	static std::mt19937 gen(std::random_device{}());

	Global::ValueType std_dev = std::sqrt(2.0 / static_cast<Global::ValueType>(prev_size));
	std::normal_distribution<> dist(0.0, std_dev);

	Global::ValueType limit = 3.0 * std_dev;

	for (auto &row : weights) {
		for (auto &w : row) {
			Global::ValueType random_value = dist(gen);
			w = std::min(std::max(random_value, -limit), limit);
			w = std::round(w * RN_ROUND_VALUE) / RN_ROUND_VALUE;
		}
	}
}

void LayerParameters::reset() {
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] = PARAM_RESET_VALUE;
		}
	}
}

void LayerParameters::add(const LayerParameters &new_gradient_layer) {
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] += new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::set(const LayerParameters &new_gradient_layer) {
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] = new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::multiply(const Global::ValueType value) {
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] *= value;
		}
	}
}

LayerParameters::LayerParameters(const LayerParameters &other)
    : weights(other.weights) {}
} // namespace nn
