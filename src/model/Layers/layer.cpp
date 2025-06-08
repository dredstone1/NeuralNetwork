#include "layer.hpp"
#include <cstddef>

namespace nn {
void Layer::add(const LayerParameters &gradients) {
	Parameters.add(gradients);
}

void Layer::reset() {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.out[i] = dots.net[i] = 0.0;
	}
}

const LayerParameters Layer::getParms() {
	return Parameters;
}

void Layer::forward(const std::vector<Global::ValueType> &metrix, const Global::ValueType) {
	for (size_t i = 0; i < Parameters.getSize(); ++i) {
		dots.net[i] = 0;

		for (size_t j = 0; j < Parameters.getPrevSize(); ++j) {
			if (j < static_cast<size_t>(metrix.size())) {
				dots.net[i] += Parameters.weights[i][j] * metrix[j];
			}
		}
		dots.out[i] = dots.net[i];
	}
}
} // namespace nn
