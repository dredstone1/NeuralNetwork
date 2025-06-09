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
} // namespace nn::model
