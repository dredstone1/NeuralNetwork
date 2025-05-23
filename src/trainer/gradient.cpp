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

gradient::gradient(const int input_size, const int output_size, const int hidden_layers_size, const int hidden_layers_count) {
	gradients.reserve(hidden_layers_count + 1);

	for (int i = 0; i < hidden_layers_count; i++) {
		if (i == 0) {
			gradients.emplace_back(LayerParameters(hidden_layers_size, input_size));
		} else {
			gradients.emplace_back(LayerParameters(hidden_layers_size, hidden_layers_size));
		}
	}

	gradients.emplace_back(LayerParameters(output_size, hidden_layers_size));
}

void gradient::multiply(const double value) {
	for (auto &gradient_layer : gradients) {
		gradient_layer.multiply(value);
	}
}

gradient::gradient(const gradient &other) : gradient(other.gradients[0].getPrevSize(), other.gradients[other.gradients.size() - 1].getSize(), other.gradients[0].getSize(), other.gradients.size() - 1) {
	gradients = other.gradients;
}
