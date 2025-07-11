#include "DenseLayer.hpp"
#include "LayerParameters.hpp"

namespace nn::model {
DenseLayer::DenseLayer(const int size, const int prevSize, const bool randomInit)
    : dots(size),
      parameters(size, prevSize),
      gradients(size, prevSize) {
	if (randomInit) {
		parameters.initializeParamRandom(getPrevSize());
	}
}

void Output_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); ++i) {
		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); ++j) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}
	}

	Activation::softmax(dots);
}

global::ParamMetrix Output_Layer::getDelta(const global::ParamMetrix &output) {
	global::ParamMetrix deltas = dots.out;
	for (size_t i = 0; i < deltas.size(); ++i) {
		deltas[i] -= output[i];
	}

	return deltas;
}

void Output_Layer::backward(
    global::ParamMetrix &deltas,
    const global::ParamMetrix &prevLayer,
    const LayerParameters *) {
	deltas = getDelta(deltas);

	for (size_t i = 0; i < getSize(); ++i) {
		gradients.bias[i] += deltas[i];

		for (size_t j = 0; j < getPrevSize(); ++j) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

global::ValueType Output_Layer::getCrossEntropyLoss(const global::ParamMetrix &prediction, const int target) {
	return -std::log(std::max(prediction[target], MIN_LOSS_VALUE));
}

global::ValueType Output_Layer::getLoss(const global::Prediction &targets) {
	return getCrossEntropyLoss(getOut(), targets.index);
}

global::ValueType Hidden_Layer::getLoss(const global::Prediction &) { return 0; }

void Hidden_Layer::forward(const global::ParamMetrix &metrix) {
	for (size_t i = 0; i < dots.size(); ++i) {
		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); ++j) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}

		dots.out[i] = activation(dots.net[i]);
	}
}

global::ParamMetrix Hidden_Layer::getDelta(
    const global::ParamMetrix &output,
    const LayerParameters &nextLayer) {
	global::ParamMetrix deltas(getSize(), 0.0);
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < nextLayer.getSize(); ++j) {
			deltas[i] += output[j] * nextLayer.weights[j][i];
		}

		deltas[i] *= derivativeActivation(dots.net[i]);
	}

	return deltas;
}

void Hidden_Layer::backward(
    global::ParamMetrix &deltas,
    const global::ParamMetrix &prevLayer,
    const LayerParameters *nextLayer) {
	if (!nextLayer)
		return;

	deltas = getDelta(deltas, *nextLayer);

	for (size_t i = 0; i < getSize(); ++i) {
		gradients.bias[i] += deltas[i];

		for (size_t j = 0; j < getPrevSize(); ++j) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

void DenseLayer::updateWeight(const global::ValueType learningRate) {
	gradients.multiply(-learningRate);
	parameters.add(gradients);
}

const global::ParamMetrix DenseLayer::getData() const {
	global::ParamMetrix matrix(getSize() * getPrevSize() + getSize(), 0);

	int currentI = 0;
	for (size_t i = 0; i < getSize(); ++i) {
		for (size_t j = 0; j < getPrevSize(); ++j) {
			matrix[currentI] = parameters.weights[i][j];

			currentI++;
		}
	}

	for (size_t i = 0; i < getSize(); ++i) {
		matrix[currentI] = parameters.bias[i];

		currentI++;
	}

	return matrix;
}

void DenseLayer::setData(const global::ParamMetrix newParam) {
	int currentI = 0;
	for (size_t i = 0; i < getSize(); ++i) {
		for (size_t j = 0; j < getPrevSize(); ++j) {
			parameters.weights[i][j] = newParam[currentI];

			currentI++;
		}
	}

	for (size_t i = 0; i < getSize(); ++i) {
		parameters.bias[i] = newParam[currentI];

		currentI++;
	}
}
} // namespace nn::model
