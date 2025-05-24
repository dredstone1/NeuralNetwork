#include "backPropagation.hpp"
#include "AiModel.hpp"
#include "gradient.hpp"
#include "model/Layers/Hidden_Layer.hpp"
#include "model/model.hpp"
#include "trainer/dataBase.hpp"
#include <cmath>
#include <vector>

using namespace std;

BackPropagation::BackPropagation(AiModel &_model) : model(_model) {}

double BackPropagation::get_cross_entropy_loss(const vector<double> &prediction, const int target) {
	return -log(prediction[target]);
}

double BackPropagation::get_total_error(const neural_network &temp_network, const int target) {
	return get_cross_entropy_loss(temp_network.layers[temp_network.config.hidden_layer_count()]->getOut(), target);
}

void BackPropagation::calculate_gradient(const Layer &layer, const vector<double> &deltas, const vector<double> &prevLayer, LayerParameters &gradient_) {
	calculate_gradient_for_weights(layer, prevLayer, deltas, gradient_);
}

vector<double> BackPropagation::calculate_delta_for_hidden(const Hidden_Layer &current_layer, const Layer &next_layer, const vector<double> &next_deltas) {
	vector<double> deltas(current_layer.getSize(), 0.0);

	for (int i = 0; i < current_layer.getSize(); i++) {
		deltas[i] = 0.0;
		for (int j = 0; j < next_layer.getSize(); j++) {
			deltas[i] += next_deltas[j] * next_layer.getWeight(j, i);
		}

		deltas[i] *= current_layer.activate_(current_layer.getDots().net[i]);
	}

	return deltas;
}

void BackPropagation::calculate_gradient_for_weights(const Layer &layer, const vector<double> &prevLayer, const vector<double> &deltas, LayerParameters &gradients) {
	for (int i = 0; i < layer.getSize(); i++) {
		for (int j = 0; j < layer.getPrevSize(); j++) {
			gradients.weights[i][j] += deltas[i] * prevLayer[j];
		}
	}
}

vector<double> BackPropagation::calculate_delta_for_output(const vector<double> &out, const int target) {
	vector<double> deltas(out);

	deltas[target] += 1.0;

	return deltas;
}

void BackPropagation::calculate_pattern_gradients(const TrainSample &sample, gradient &_gradients, const neural_network &temp_network) {
	vector<double> deltas;

	for (int layer_index = _gradients.gradients.size() - 1; layer_index >= 0; layer_index--) {
		Layer &layer = *temp_network.layers[layer_index];

		if (layer.getType() == OUTPUT) {
			deltas = calculate_delta_for_output(layer.getOut(), sample._prediction.index);
		} else {
			deltas = calculate_delta_for_hidden((Hidden_Layer &)layer, *temp_network.layers[layer_index + 1], deltas);
		}

		if (layer_index == 0) {
			calculate_gradient(layer, deltas, sample.input, _gradients.gradients[layer_index]);
		} else {
			calculate_gradient(layer, deltas, temp_network.layers[layer_index - 1]->getOut(), _gradients.gradients[layer_index]);
		}
	}
}

double BackPropagation::run_back_propagation(const TrainSample &sample, gradient &local_gradient) {
	model.run_model(sample.input);
	calculate_pattern_gradients(sample, local_gradient, model._model->network);

	return get_total_error(model._model->network, sample._prediction.index);
}

double BackPropagation::run_back_propagation(const Batch &batch, const double learning_rate) {
	const int batch_size = batch.size();
	double error = 0.0;

	gradient batch_gradient(model.config->network_config);
	for (int i = 0; i < batch_size; i++) {
		error += run_back_propagation(*batch.samples.at(i), batch_gradient);
	}

	update_weights(batch_size, batch_gradient, learning_rate);
	return error / batch_size;
}

void BackPropagation::update_weights(int batch_size, gradient &gradients, double learning_rate) {
	gradients.multiply(-learning_rate / batch_size);
	model._model->updateWeights(gradients);
}
