#include "Output_Layer.hpp"
#include "../activations.hpp"

namespace nn::model {
void Output_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = 0;

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}
	}

	Activation::softmax(dots);
}
} // namespace nn::model
