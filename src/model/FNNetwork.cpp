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

void DenseNetwork::fromJson(const nlohmann::json &j) {
	int inputSize = j.at("input_size");
	int pastLayerSize = inputSize;

	const auto &layerConfigs = j.at("layers");
	for (const auto &layerJson : layerConfigs) {
		int size = layerJson.at("size");
		global::ValueType weights_init_value = layerJson.at("weights_init_value");
		ActivationType AT = layerJson.at("AT");

		layers.push_back(std::make_unique<Hidden_Layer>(
		    size,
		    pastLayerSize,
		    AT));
		pastLayerSize = size;
	}

	int outputSize = j.at("output_size");
	global::ValueType output_init_value = j.at("output_init_value");
	layers.push_back(std::make_unique<Output_Layer>(
	    outputSize,
	    pastLayerSize));
}

void DenseNetwork::resetGradient() {

}

int DenseNetwork::
} // namespace nn::model
