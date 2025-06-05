#include "Hidden_Layer.hpp"

void Hidden_Layer::forward(const std::vector<Global::ValueType> &metrix) {
	for (int i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;
		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += Parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = activate_(dots.net[i]);
	}
}
