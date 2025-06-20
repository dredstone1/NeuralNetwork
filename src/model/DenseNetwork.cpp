#include "DenseNetwork.hpp"
#include "Globals.hpp"

namespace nn::model {
void DenseNetwork::forward(const global::ParamMetrix &input) {
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
	}
}

void DenseNetwork::backword(const global::ParamMetrix &output) {
	global::ParamMetrix deltas;
	layers[layers.size() - 1]->backword(output, deltas);

	for (size_t i = layers.size() - 2; i >= 0; i--) {
		global::ParamMetrix tempDeltas = deltas;
		layers[i]->backword(tempDeltas, deltas);
	}
}
} // namespace nn::model
