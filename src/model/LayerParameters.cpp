#include "LayerParameters.hpp"
#include <cmath>
#include <random>
#include <vector>

LayerParameters::LayerParameters(const int size, const int prev_size, const double init_value) {
	weights.resize(size, std::vector<double>(prev_size, init_value));
	if (init_value >= 0.0)
		return;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist(0.0, sqrt(2.0 / (prev_size)));

	for (auto &dot : weights) {
		for (auto &weight : dot) {
			double random_value = dist(gen);
			random_value = fmin(fmax(random_value, -1.0), 1.0);
			random_value = round(random_value * 10000.0) / 10000.0;
			weight = random_value;
		}
	}
}

void LayerParameters::reset() {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] = 0.0;
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

void LayerParameters::multiply(const double value) {
	for (int i = 0; i < getSize(); i++) {
		for (int j = 0; j < getPrevSize(); j++) {
			weights[i][j] *= value;
		}
	}
}

LayerParameters::LayerParameters(const LayerParameters &other) {
	weights = other.weights;
}
