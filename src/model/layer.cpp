#include "layer.hpp"
#include "Globals.hpp"

namespace nn::model {
void Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < parameters.getSize(); ++i) {
		dots.net[i] = 0;

		for (size_t j = 0; j < parameters.getPrevSize(); ++j) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = dots.net[i];
	}
}

void Output_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}
	}

	Activation::softmax(dots);
}

void Hidden_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = activation(dots.net[i]);
	}
}
} // namespace nn::model
