#include "DenseLayer.hpp"
#include <random>

namespace nn::model::fnn {
DenseLayer::DenseLayer(
    const int size,
    const int prevSize,
    const ActivationType activation,
    const bool randomInit)
    : dots(size),
      parameters(size, prevSize),
      gradients(size, prevSize),
      activationFunction(activation) {
	if (randomInit) {
		parameters.initializeParamRandom(getPrevSize());
	}
}

void Hidden_Layer::CreateDropoutMask() {
	const float keepProb = 1.0f - config.dropoutRate;

	if (dropoutMask.size() != dots.size()) {
		dropoutMask.resize(dots.size());
	}

	static thread_local std::mt19937 rng{std::random_device{}()};
	std::bernoulli_distribution bernoulli(keepProb);

	for (size_t i = 0; i < dropoutMask.size(); ++i) {
		dropoutMask[i] = static_cast<uint8_t>(bernoulli(rng));
	}
}

void Output_Layer::forward(const global::ParamMetrix &metrix) {

	for (size_t i = 0; i < dots.size(); ++i) {
		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); ++j) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}
	}

	activationFunction.activate(dots.net, dots.out);
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
	if (activationFunction.getType() == ActivationType::Softmax) {
		deltas = getDelta(deltas);
	} else {
		activationFunction.derivativeActivate(dots.out, deltas);
	}

	for (size_t i = 0; i < getSize(); ++i) {
		gradients.bias[i] += deltas[i];

		for (size_t j = 0; j < getPrevSize(); ++j) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

global::ValueType Output_Layer::getCrossEntropyLoss(
    const global::ParamMetrix &prediction,
    const int target) {
	return -std::log(std::max(prediction[target], MIN_LOSS_VALUE));
}

global::ValueType Output_Layer::getLoss(const global::Prediction &targets) {
	return getCrossEntropyLoss(getOut(), targets.index);
}

void Hidden_Layer::forward(const global::ParamMetrix &metrix) {
	if (isTraining) {
		CreateDropoutMask();
	}
	const float keepProb = 1.0f - config.dropoutRate;

	for (size_t i = 0; i < dots.size(); ++i) {
		if (isTraining && dropoutMask[i] == 0) {
			dots.net[i] = 0;
			continue;
		}

		dots.net[i] = parameters.bias[i];

		for (size_t j = 0; j < metrix.size(); ++j) {
			dots.net[i] += parameters.weights[i][j] * metrix[j];
		}

		if (isTraining) {
			dots.net[i] /= keepProb;
		}
	}

	activationFunction.activate(dots.net, dots.out);
}

global::ParamMetrix Hidden_Layer::getDelta(
    const global::ParamMetrix &output,
    const LayerParameters &nextLayer) {
	global::ParamMetrix deltas(getSize(), 0.0);
	for (size_t i = 0; i < getSize(); i++) {
		for (size_t j = 0; j < nextLayer.getSize(); ++j) {
			deltas[i] += output[j] * nextLayer.weights[j][i];
		}
	}

	activationFunction.derivativeActivate(dots.out, deltas);

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
		if (isTraining && dropoutMask[i] == 0) {
			deltas[i] = 0;
			continue;
		}

		gradients.bias[i] += deltas[i];

		for (size_t j = 0; j < getPrevSize(); ++j) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

void DenseLayer::updateWeight(const std::shared_ptr<nn::model::IOptimizer> optimizer) {
	optimizer->step(parameters.bias.data(), gradients.bias.data(), parameters.bias.size());

	for (size_t i = 0; i < parameters.weights.size(); ++i) {
		optimizer->step(
		    parameters.weights[i].data(),
		    gradients.weights[i].data(),
		    parameters.weights[i].size());
	}
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

void Neurons::reset() {
	for (size_t i = 0; i < out.size(); ++i) {
		out[i] = 0.0;
		net[i] = 0.0;
	}
}

Neurons::Neurons(const int size) {
	out.resize(size, 0.0);
	net.resize(size, 0.0);
}
} // namespace nn::model::fnn
