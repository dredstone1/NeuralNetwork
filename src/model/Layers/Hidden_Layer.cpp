#include "Hidden_Layer.hpp"
#include "../activations.hpp"

void Hidden_Layer::forward(const vector<double> &metrix) {
	for (int i = 0; i < dots.size(); i++) {
		// dots.net[i] = Parameters->bias[i];

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += Parameters->weights[i][j] * metrix[j];
		}

		dots.out[i] = ActivationFunctions::LeakyRelu(dots.net[i]);
	}
}
