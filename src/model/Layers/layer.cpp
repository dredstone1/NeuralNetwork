#include "layer.hpp"

void Layer::add(const LayerParameters &gradients) {
	Parameters.add(gradients);
}

void Layer::reset() {
	for (int i = 0; i < dots.size(); i++) {
		dots.out[i] = dots.net[i] = 0.0;
	}
}

const LayerParameters Layer::getParms() {
	return Parameters;
}

void Layer::forward(const std::vector<double> &metrix) {
	for (int i = 0; i < Parameters.getSize(); ++i) {
		dots.net[i] = 0;

		for (int j = 0; j < Parameters.getPrevSize(); ++j) {
			if (j < static_cast<int>(metrix.size())) {
				dots.net[i] += Parameters.weights[i][j] * metrix[j];
			}
		}
		dots.out[i] = dots.net[i];
	}
}
