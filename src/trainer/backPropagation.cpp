#include "backPropagation.hpp"
#include <cmath>

namespace nn::training {
BackPropagation::BackPropagation(AiModel &_model, learningRateParams &learningRate_)
    : local_gradient(_model.getConfig().config_data.network_config),
      model(_model),
      learningRate(learningRate_) {}

global::ValueType BackPropagation::get_cross_entropy_loss(const global::ParamMetrix &prediction, const int target) {
	return -std::log(std::max(prediction[target], MIN_LOSS_VALUE));
}

global::ValueType BackPropagation::get_total_error(const model::NeuralNetwork &temp_network, const int target) {
	return get_cross_entropy_loss(temp_network.layers[temp_network.config.hidden_layer_count()]->getOut(), target);
}

void BackPropagation::calculate_gradient(
    const model::Layer &layer,
    const global::ParamMetrix &deltas,
    const global::ParamMetrix &prevLayer,
    model::LayerParameters &gradient_) {
	calculateGradientForWeights(layer, prevLayer, deltas, gradient_);
	calculateGradientForBiases(layer, deltas, gradient_);
}

global::ParamMetrix BackPropagation::calculateDeltaHidden(
    const model::Hidden_Layer &current_layer,
    const model::Layer &next_layer,
    const global::ParamMetrix &next_deltas) {
	global::ParamMetrix deltas(current_layer.getSize(), 0.0);

	for (size_t i = 0; i < current_layer.getSize(); i++) {
		deltas[i] = 0.0;

		for (size_t j = 0; j < next_layer.getSize(); j++) {
			deltas[i] += next_deltas[j] * next_layer.getWeight(j, i);
		}

		deltas[i] *= current_layer.derivativeActivation(current_layer.getDots().net[i]);
	}

	return deltas;
}

void BackPropagation::calculateGradientForWeights(
    const model::Layer &layer,
    const global::ParamMetrix &prevLayer,
    const global::ParamMetrix &deltas,
    model::LayerParameters &gradients) {
	for (size_t i = 0; i < layer.getSize(); i++) {
		for (size_t j = 0; j < layer.getPrevSize(); j++) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

void BackPropagation::calculateGradientForBiases(
    const model::Layer &layer,
    const global::ParamMetrix &deltas,
    model::LayerParameters &gradients) {
	for (size_t i = 0; i < layer.getSize(); i++) {
		gradients.bias[i] += deltas[i];
	}
}

global::ParamMetrix BackPropagation::calculateDeltaForOutput(const global::ParamMetrix &out, const int target) {
	global::ParamMetrix deltas(out);

	deltas[target] -= 1;

	return deltas;
}

void BackPropagation::calculate_pattern_gradients(const TrainSample &sample, const model::NeuralNetwork &temp_network) {
	global::ParamMetrix deltas;

	for (int layer_index = local_gradient.gradients.size() - 1; layer_index >= 0; layer_index--) {
		const model::Layer &layer = *temp_network.layers.at(layer_index);

		if (layer.getType() == model::LayerType::OUTPUT) {
			deltas = calculateDeltaForOutput(layer.getOut(), sample.prediction.index);
		} else {
			deltas = calculateDeltaHidden((model::Hidden_Layer &)layer, *temp_network.layers[layer_index + 1], deltas);
		}

		if (layer_index == 0) {
			calculate_gradient(layer, deltas, sample.input, local_gradient.gradients[layer_index]);
		} else {
			calculate_gradient(layer, deltas, temp_network.layers[layer_index - 1]->getOut(), local_gradient.gradients[layer_index]);
		}
	}
}

global::ValueType BackPropagation::run_back_propagation(const TrainSample &sample) {
	model.model->runModel(sample.input);
	calculate_pattern_gradients(sample, model.model->network);

	return get_total_error(model.model->network, sample.prediction.index);
}

global::ValueType BackPropagation::run_back_propagation(const Batch &batch) {
	global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	local_gradient.reset();
	for (size_t i = 0; i < batch.size(); i++) {
		const TrainSample *current_sample_ptr = batch.samples.at(i);

		model.model->visual.updatePrediction(current_sample_ptr->prediction.index);
		error += run_back_propagation(*current_sample_ptr);
	}

	update_weights(batch.size(), learningRate.currentLearningRate);
	return error / batch.size();
}

void BackPropagation::update_weights(int batch_size, global::ValueType learning_rate) {
	local_gradient.multiply(-learning_rate / batch_size);
	model.model->updateWeights(local_gradient);
}
} // namespace nn::training
