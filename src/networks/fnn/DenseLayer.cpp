#include "DenseLayer.hpp"
#include <cstddef>
#include <random>
#include <vector>

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
      deltaL({size}),
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

	static std::vector<global::ValueType> temp(dropoutMask.numElements(), 0);
	for (size_t i = 0; i < dropoutMask.numElements(); ++i) {
		temp[i] = static_cast<uint8_t>(bernoulli(rng));
	}

	dropoutMask = temp;
}

void Output_Layer::forward(const global::Tensor &metrix) {
	parameters.weights.matmul(metrix, net);
	net += parameters.biases;

	activationFunction.activate(net, out);
}

void Output_Layer::getDelta(const global::Tensor &output) {
	deltaL = out;
	deltaL -= output;
}

void Output_Layer::backward(
    global::Tensor **deltas,
    const global::Tensor &prevLayer,
    const LayerParams *) {
	if (activationFunction.getType() == ActivationType::Softmax) {
		getDelta(**deltas);
	} else {
		activationFunction.derivativeActivate(out, **deltas);
		deltaL = **deltas;
	}

	gradients.biases += deltaL;
	global::Tensor::outer(deltaL, prevLayer, gradients.weights);
	*deltas = &deltaL;
}

global::ValueType Output_Layer::getCrossEntropyLoss(
    const global::Tensor &prediction,
    const size_t target) {
	return -std::log(std::max(prediction.getValue({target}), MIN_LOSS_VALUE));
}

global::ValueType Output_Layer::getLoss(const global::Prediction &targets) {
	return getCrossEntropyLoss(getOut(), targets.index);
}

void Hidden_Layer::forward(const global::Tensor &metrix) {
	if (isTraining)
		CreateDropoutMask();

	parameters.weights.matmul(metrix, net);
	net += parameters.biases;

	if (isTraining && config.dropoutRate > 0.0f) {
		net /= 1.0f - config.dropoutRate;
		net *= dropoutMask;
	}

	activationFunction.activate(net, out);
}

void Hidden_Layer::backward(
    global::Tensor **deltas,
    const global::Tensor &prevLayer,
    const LayerParams *nextLayer) {

	if (!nextLayer)
		return;

	calculateDelta(**deltas, *nextLayer);

	if (isTraining && config.dropoutRate) {
		deltaL *= dropoutMask;
	}

	gradients.biases += deltaL;

	global::Tensor::outer(deltaL, prevLayer, gradients.weights);
	*deltas = &deltaL;
}

void Hidden_Layer::calculateDelta(
    const global::Tensor &output,
    const LayerParams &nextLayer) {

	nextLayer.weights.matmulT(output, deltaL);
	activationFunction.derivativeActivate(out, deltaL);
}

size_t DenseLayer::getParamCount() const {
	return size() * prevSize() + size();
}

void DenseLayer::updateWeight(nn::model::IOptimizer &optimizer) {
	optimizer.step(parameters.biases, gradients.biases);
	optimizer.step(parameters.weights, gradients.weights);
}

const global::Tensor DenseLayer::getData() const {
	size_t weightsSize = parameters.weights.numElements();
	size_t biasesSize = parameters.biases.numElements();

	global::Tensor matrix({weightsSize + biasesSize});

	// Copy weights
	matrix.insertRange(parameters.weights, 0, 0, weightsSize);

	// Copy biases
	matrix.insertRange(parameters.biases, 0, weightsSize, biasesSize);

	return matrix;
}

void DenseLayer::setData(const global::Tensor newParam, const size_t offset) {
	size_t weightsSize = parameters.weights.numElements();
	size_t biasesSize = parameters.biases.numElements();

	// Copy into weights
	parameters.weights.insertRange(newParam, offset, 0, weightsSize);

	// Copy into biases
	parameters.biases.insertRange(newParam, offset + weightsSize, 0, biasesSize);
}

void DenseLayer::fillParamRandom() {
	static std::mt19937 gen(std::random_device{}());

	global::ValueType std_dev = std::sqrt(2.0 / static_cast<global::ValueType>(prevSize()));
	std::normal_distribution<> dist(0.0, std_dev);

	std::vector<global::ValueType> temp(parameters.weights.numElements());
	for (size_t i = 0; i < temp.size(); ++i) {
		temp[i] = dist(gen);
	}
	parameters.weights = temp;
}

void DenseLayer::resetDots() {
	net.zero();
	out.zero();
}

void DenseLayer::resetGradient() {
	gradients.biases.zero();
	gradients.weights.zero();
}
} // namespace nn::model::fnn
