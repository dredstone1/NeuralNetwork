#include "Output_Layer.hpp"
#include "../activations.hpp"


void Output_Layer::forward(const std::vector<double> &metrix) {
	for (int i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += Parameters.weights[i][j] * metrix[j];
		}
	}

    activations::Softmax(dots);
}
