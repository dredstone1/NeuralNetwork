#include "backPropagation.hpp"
#include "Globals.hpp"
#include "gradient.hpp"
#include "trainer/dataBase.hpp"
#include "trainer/learning_rate.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace nn {
BackPropagation::BackPropagation(AiModel &_model, LearningRate &lr_)
    : local_gradient(_model.getConfig().config_data.network_config),
      model(_model),
      lr(lr_) {}

Global::ValueType BackPropagation::get_cross_entropy_loss(const std::vector<Global::ValueType> &prediction, const int target) {
	return -std::log(std::max(prediction[target], MIN_LOSS_VALUE));
}

Global::ValueType BackPropagation::get_total_error(const neural_network &temp_network, const int target) {
	return get_cross_entropy_loss(temp_network.layers[temp_network.config.hidden_layer_count()]->getOut(), target);
}

void BackPropagation::calculate_gradient(const Layer &layer, const std::vector<Global::ValueType> &deltas, const std::vector<Global::ValueType> &prevLayer, LayerParameters &gradient_) {
	calculate_gradient_for_weights(layer, prevLayer, deltas, gradient_);
}

std::vector<Global::ValueType> BackPropagation::calculate_delta_for_hidden(
    const Hidden_Layer &current_layer,
    const Layer &next_layer,
    const std::vector<Global::ValueType> &next_deltas) {
	std::vector<Global::ValueType> deltas(current_layer.getSize(), 0.0);

	for (size_t i = 0; i < current_layer.getSize(); i++) {
		deltas[i] = 0.0;
		for (size_t j = 0; j < next_layer.getSize(); j++) {
			deltas[i] += next_deltas[j] * next_layer.getWeight(j, i);
		}

		deltas[i] *= current_layer.Derivative_activate(current_layer.getDots().net[i]);
	}

	return deltas;
}
Global::ValueType BackPropagation::clippValue(const Global::ValueType value) {
	// if (std::abs(value) < 0.0001)
	// 	return 0.0001;
	// if (std::abs(value) > 1)
	// 	return 1;
	return value;
}

void BackPropagation::calculate_gradient_for_weights(const Layer &layer, const std::vector<Global::ValueType> &prevLayer, const std::vector<Global::ValueType> &deltas, LayerParameters &gradients) {
	for (size_t i = 0; i < layer.getSize(); i++) {
		for (size_t j = 0; j < layer.getPrevSize(); j++) {
			gradients.weights[i][j] += clippValue(deltas[i] * prevLayer[j]);
		}
	}
}

std::vector<Global::ValueType> BackPropagation::calculate_delta_for_output(const std::vector<Global::ValueType> &out, const int target) {
	std::vector<Global::ValueType> deltas(out);

	deltas[target] -= 1;

	return deltas;
}

void BackPropagation::addNoise(std::vector<Global::ValueType> &input, Global::ValueType noise_level) {
	if (noise_level <= 0) {
		return;
	}

	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-noise_level, noise_level);

	for (auto &val : input) {
		val += dis(gen);
		if (val < 0.0f)
			val = 0.0f;
		if (val > 1.0f)
			val = 1.0f;
	}
}

void BackPropagation::calculate_pattern_gradients(const TrainSample &sample, const neural_network &temp_network) {
	std::vector<Global::ValueType> deltas;

	for (int layer_index = local_gradient.gradients.size() - 1; layer_index >= 0; layer_index--) {
		const Layer &layer = *temp_network.layers.at(layer_index);

		if (layer.getType() == OUTPUT) {
			deltas = calculate_delta_for_output(layer.getOut(), sample._prediction.index);
		} else {
			deltas = calculate_delta_for_hidden((Hidden_Layer &)layer, *temp_network.layers[layer_index + 1], deltas);
		}

		if (layer_index == 0) {
			calculate_gradient(layer, deltas, sample.input, local_gradient.gradients[layer_index]);
		} else {
			calculate_gradient(layer, deltas, temp_network.layers[layer_index - 1]->getOut(), local_gradient.gradients[layer_index]);
		}
	}
}

Global::ValueType BackPropagation::run_back_propagation(const TrainSample &sample) {
	model._model->run_model(sample.input, model.getConfig().config_data.training_config.drop_out_rate);
	calculate_pattern_gradients(sample, model._model->network);

	return get_total_error(model._model->network, sample._prediction.index);
}

Global::ValueType BackPropagation::run_back_propagation(const Batch &batch) {
	Global::ValueType error = 0.0;

	if (batch.size() == 0) {
		return error;
	}

	local_gradient.reset();
	for (size_t i = 0; i < batch.size(); i++) {
		const TrainSample *current_sample_ptr = batch.samples.at(i);
		TrainSample new_sample = *current_sample_ptr;
		addNoise(new_sample.input, model.config.config_data.training_config.noise_level);

		model._model->visual.update_prediction(new_sample._prediction.index);
		error += run_back_propagation(new_sample);
	}

	update_weights(batch.size(), lr.getLearningRate());
	return error / batch.size();
}

void BackPropagation::update_weights(int batch_size, Global::ValueType learning_rate) {
	local_gradient.multiply(-learning_rate / batch_size);
	model._model->updateWeights(local_gradient);
}
} // namespace nn
