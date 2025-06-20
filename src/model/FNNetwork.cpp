#include "FNNetwork.hpp"
#include "DenseLayer.hpp"
#include "Globals.hpp"
#include "LayerParameters.hpp"
#include <memory>
#include <nlohmann/json_fwd.hpp>

namespace nn::model {
void DenseNetwork::forward(const global::ParamMetrix &input) {
	layers[0]->forward(input);

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->forward(layers[i - 1]->getOut());
	}
}

void DenseNetwork::backword(const global::ParamMetrix &output, global::ParamMetrix &deltas) {
	if (layers.size() > 1) {
		layers[layers.size() - 1]->backword(output, deltas, layers[layers.size() - 2]->getOut(), LayerParameters(0, 0));
	} else {
		layers[layers.size() - 1]->backword(output, deltas, layers[layers.size() - 1]->getNet(), LayerParameters(0, 0));
	}

	for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
		global::ParamMetrix tempDeltas = deltas;

		if (i > 0) {
			layers[i]->backword(tempDeltas, deltas, layers[i - 1]->getOut(), layers[i + 1]->getParms());
		} else {
			layers[i]->backword(tempDeltas, deltas, layers[i]->getNet(), layers[i + 1]->getParms());
		}
	}
}

global::ValueType DenseNetwork::getLost(const global::ParamMetrix &output) const {
	return layers[layers.size() - 1]->getLost(output);
}

void DenseNetwork::resetGradient() {
	for (auto &layer : layers) {
		layer->resetGradient();
	}
}

int DenseNetwork::inputSize() const {
	return layers[0]->getPrevSize();
}

int DenseNetwork::outputSize() const {
	return layers[layers.size() - 1]->getSize();
}

const global::ParamMetrix &DenseNetwork::getOutput() const {
	return layers[layers.size() - 1]->getOut();
}
} // namespace nn::model
