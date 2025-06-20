#include "DenseLayer.hpp"
#include "Globals.hpp"
#include "LayerParameters.hpp"
#include "activations.hpp"

namespace nn::model {
void Output_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}
	}

	Activation::softmax(dots);
}

global::ParamMetrix Output_Layer::getDelta(const global::ParamMetrix &output) {
	global::ParamMetrix deltas(dots.out);
	for (size_t i = 0; i < output.size(); i++) {
		deltas[i] -= output[i];
	}

	return deltas;
}

void Output_Layer::backword(
    const global::ParamMetrix &output,
    global::ParamMetrix &newDeltas,
    const global::ParamMetrix &prevLayer,
    const LayerParameters &) {
	newDeltas = getDelta(output);

	for (size_t i = 0; i < getSize(); i++) {
		gradients.bias[i] += newDeltas[i];

		for (size_t j = 0; j < getPrevSize(); j++) {
			gradients.weights[i][j] += newDeltas[i] * prevLayer[j];
		}
	}
}

global::ValueType Output_Layer::get_cross_entropy_loss(const global::ParamMetrix &prediction, const int target) {
	return -std::log(std::max(prediction[target], MIN_LOSS_VALUE));
}

global::ValueType Output_Layer::getLost(const global::ParamMetrix &output) {
	int max_value = 0;
	for (size_t i = 0; i < output.size(); i++) {
		if (output[i] > output[max_value]) {
			max_value = i;
		}
	}

	return get_cross_entropy_loss(getOut(), max_value);
}

global::ValueType Hidden_Layer::getLost(const global::ParamMetrix &) {
	return 1;
}

void Hidden_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); i++) {
		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); j++) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = activation(dots.net[i]);
	}
}

global::ParamMetrix Hidden_Layer::getDelta(const global::ParamMetrix &output, const LayerParameters &nextLayer) {
	global::ParamMetrix deltas(getSize(), 0.0);
	for (size_t i = 0; i < getSize(); i++) {
		deltas[i] = 0.0;

		for (size_t j = 0; j < nextLayer.getSize(); j++) {
			deltas[i] += output[j] * nextLayer.weights[i][j];
		}

		deltas[i] *= derivativeActivation(getDots().net[i]);
	}

	return deltas;
}

void Hidden_Layer::backword(
    const global::ParamMetrix &output,
    global::ParamMetrix &newDeltas,
    const global::ParamMetrix &prevLayer,
    const LayerParameters &nextLayer) {
	newDeltas = getDelta(output, nextLayer);

	for (size_t i = 0; i < getSize(); i++) {
		gradients.bias[i] += newDeltas[i];

		for (size_t j = 0; j < getPrevSize(); j++) {
			gradients.weights[i][j] += newDeltas[i] * prevLayer[j];
		}
	}
}

void Hidden_Layer::updateWeight(const global::ValueType learningRate) {
	gradients.multiply(learningRate);
	parameters.add(gradients);
}

void Output_Layer::updateWeight(const global::ValueType learningRate) {
	gradients.multiply(learningRate);
	parameters.add(gradients);
}
} // namespace nn::model
