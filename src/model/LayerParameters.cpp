#include "LayerParameters.hpp"
#include <cmath>
#include <random>
#include <vector>

LayerParameters::LayerParameters(const int size, const int prev_size, const double init_value) {
	weights.resize(size, std::vector<double>(prev_size, init_value));

	if (init_value < 0.0)
		initialize_Param_rn(prev_size);
}

void LayerParameters::initialize_Param_rn(const int prev_size) {
	static std::mt19937 gen(std::random_device{}());

	double std_dev = std::sqrt(2.0 / static_cast<double>(prev_size));
	std::normal_distribution<> dist(0.0, std_dev);

	double limit = 3.0 * std_dev;

	for (auto &row : weights) {
		for (auto &w : row) {
			double random_value = dist(gen);
			w = std::min(std::max(random_value, -limit), limit);
			w = std::round(w * RN_ROUND_VALUE) / RN_ROUND_VALUE;
		}
	}
}

void LayerParameters::reset() {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] = PARAM_RESET_VALUE;
		}
	}
}

void LayerParameters::add(const LayerParameters &new_gradient_layer) {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] += new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::set(const LayerParameters &new_gradient_layer) {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] = new_gradient_layer.weights[i][j];
		}
	}
}

void LayerParameters::multiply(const double value) {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] *= value;
		}
	}
}

LayerParameters::LayerParameters(const LayerParameters &other)
    : weights(other.weights) {}
