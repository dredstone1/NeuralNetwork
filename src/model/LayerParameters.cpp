#include "LayerParameters.hpp"
#include "Globals.hpp"
#include <random>

namespace nn::model {
LayerParameters::LayerParameters(const int size, const int prev_size) {
	weights.resize(size, global::ParamMetrix(prev_size, 0.1));
	bias.resize(size, 0.0001);
}

void LayerParameters::initializeParamRandom(const int prev_size) {
	static std::mt19937 gen(std::random_device{}());

	global::ValueType std_dev = std::sqrt(2.0 / static_cast<global::ValueType>(prev_size));
	std::normal_distribution<> dist(0.0, std_dev);

	global::ValueType limit = 3.0 * std_dev;

	for (auto &row : weights) {
		for (auto &w : row) {
			global::ValueType random_value = dist(gen);
			w = std::min(std::max(random_value, -limit), limit);
			w = std::round(w * RN_ROUND_VALUE) / RN_ROUND_VALUE;
		}
	}
}

void LayerParameters::reset() {
	for (size_t i = 0; i < getSize(); i++) {
		bias[i] = PARAM_RESET_VALUE;

		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] = PARAM_RESET_VALUE;
		}
	}
}

void LayerParameters::add(const LayerParameters &new_gradient_layer) {
	for (size_t i = 0; i < getSize(); i++) {
		bias[i] += new_gradient_layer.bias[i];

		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] += new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::set(const LayerParameters &new_gradient_layer) {
	for (size_t i = 0; i < getSize(); i++) {
        bias[i] = new_gradient_layer.bias[i];
		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] = new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::multiply(const global::ValueType value) {
	for (size_t i = 0; i < getSize(); i++) {
		bias[i] *= value;

		for (size_t j = 0; j < getPrevSize(); j++) {
			weights[i][j] *= value;
		}
	}
}
} // namespace nn::model
