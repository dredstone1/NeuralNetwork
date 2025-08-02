#include "DenseLayer.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <random>

namespace nn::model::fnn {
DenseLayer::DenseLayer(
    const size_t size,
    const size_t prevSize,
    const ActivationType activation,
    const bool randomInit)
    : net({size}),
      out({size}),
      parameters(size, prevSize),
      gradients(size, prevSize),
      activationFunction(activation) {
	if (randomInit) {
		fillParamRandom();
	}
}

void Hidden_Layer::CreateDropoutMask() {
	if (config.dropoutRate == 0) {
		return;
	}

	const float keepProb = 1.0f - config.dropoutRate;

	static thread_local std::mt19937 rng{std::random_device{}()};
	std::bernoulli_distribution bernoulli(keepProb);

	for (size_t i = 0; i < dropoutMask.numElements(); ++i) {
		dropoutMask({i}) = static_cast<uint8_t>(bernoulli(rng));
	}
}

void Output_Layer::forward(const global::Tensor &metrix) {
	for (size_t i = 0; i < size(); ++i) {
		net({i}) = parameters.biases({i});

		for (size_t j = 0; j < metrix.numElements(); ++j) {
			net({i}) += parameters.weights({i, j}) * metrix({j});
		}
	}

	activationFunction.activate(net, out);
}

global::Tensor Output_Layer::getDelta(const global::Tensor &output) {
	global::Tensor deltas = out;
	for (size_t i = 0; i < size(); ++i) {
		deltas({i}) -= output({i});
	}

	return deltas;
}

void Output_Layer::backward(
    global::Tensor &deltas,
    const global::Tensor &prevLayer,
    const LayerParams *) {
	if (activationFunction.getType() == ActivationType::Softmax) {
		deltas = getDelta(deltas);
	} else {
		activationFunction.derivativeActivate(out, deltas);
	}

	gradients.biases += deltas;
	for (size_t i = 0; i < size(); ++i) {

		for (size_t j = 0; j < prevSize(); ++j) {
			gradients.weights({i, j}) += deltas({i}) * prevLayer({j});
		}
	}
}

global::ValueType Output_Layer::getCrossEntropyLoss(
    const global::Tensor &prediction,
    const size_t target) {
	return -std::log(std::max(prediction({target}), MIN_LOSS_VALUE));
}

global::ValueType Output_Layer::getLoss(const global::Prediction &targets) {
	return getCrossEntropyLoss(getOut(), targets.index);
}

void Hidden_Layer::forward(const global::Tensor &metrix) {
	if (isTraining) {
		CreateDropoutMask();
	}
	const float keepProb = 1.0f - config.dropoutRate;

	for (size_t i = 0; i < size(); ++i) {
		if (isTraining && config.dropoutRate && dropoutMask({i}) == 0) {
			net({i}) = 0;
			continue;
		}

		net({i}) = parameters.biases({i});

		for (size_t j = 0; j < metrix.numElements(); ++j) {
			net({i}) += parameters.weights({i, j}) * metrix({j});
		}

		if (isTraining) {
			net({i}) /= keepProb;
		}
	}

	activationFunction.activate(net, out);
}

global::Tensor Hidden_Layer::getDelta(
    const global::Tensor &output,
    const LayerParams &nextLayer) {
	global::Tensor deltas({size()});
	for (size_t i = 0; i < size(); i++) {
		for (size_t j = 0; j < nextLayer.size(); ++j) {
			deltas({i}) += output({j}) * nextLayer.weights({j, i});
		}
	}

	activationFunction.derivativeActivate(out, deltas);

	return deltas;
}

void Hidden_Layer::backward(
    global::Tensor &deltas,
    const global::Tensor &prevLayer,
    const LayerParams *nextLayer) {

	if (!nextLayer)
		return;

	deltas = getDelta(deltas, *nextLayer);

	if (isTraining && config.dropoutRate) {
		deltas *= dropoutMask;
	}
	gradients.biases += deltas;

	for (size_t i = 0; i < size(); ++i) {

		for (size_t j = 0; j < prevSize(); ++j) {
			gradients.weights({i, j}) += deltas({i}) * prevLayer({j});
		}
	}
}

size_t DenseLayer::getParamCount() const {
	return size() * prevSize() + size();
}

void DenseLayer::updateWeight(nn::model::IOptimizer &optimizer) {
	optimizer.step(parameters.biases, gradients.biases);

	for (size_t i = 0; i < size(); ++i) {
		optimizer.step(parameters.weights, gradients.weights);
	}
}

const global::Tensor DenseLayer::getData() const {
	global::Tensor matrix({parameters.paramSize()});

	size_t currentI = 0;
	for (size_t i = 0; i < size(); ++i) {
		for (size_t j = 0; j < prevSize(); ++j) {
			matrix({currentI}) = parameters.weights({i, j});

			++currentI;
		}
	}

	for (size_t i = 0; i < size(); ++i) {
		matrix({currentI}) = parameters.biases({i});

		++currentI;
	}

	return matrix;
}

void DenseLayer::setData(const global::Tensor newParam) {
	size_t currentI = 0;
	for (size_t i = 0; i < size(); ++i) {
		for (size_t j = 0; j < prevSize(); ++j) {
			parameters.weights({i, j}) = newParam({currentI});

			++currentI;
		}
	}

	for (size_t i = 0; i < size(); ++i) {
		parameters.biases({i}) = newParam({currentI});

		++currentI;
	}
}

void DenseLayer::fillParamRandom() {
	static std::mt19937 gen(std::random_device{}());

	global::ValueType std_dev = std::sqrt(2.0 / static_cast<global::ValueType>(prevSize()));
	std::normal_distribution<> dist(0.0, std_dev);

	for (auto &value : parameters.weights) {
		value = dist(gen);
	}
}

void DenseLayer::resetDots() {
	for (size_t i = 0; i < net.numElements(); ++i) {
		net({i}) = 0;
		out({i}) = 0;
	}
}

void DenseLayer::resetGradient() {
	for (auto &value : gradients.biases) {
		value = 0;
	}

	for (auto &value : gradients.weights) {
		value = 0;
	}
}
} // namespace nn::model::fnn
