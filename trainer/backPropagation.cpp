#include "backPropagation.hpp"
#include "../model/activations.hpp"
#include "gradient.hpp"
#include <cmath>
#include <vector>

using namespace std;

BackPropagation::BackPropagation(AiModel &_model) : model(_model), global_gradient(gradient(_model._model->network.input_size, _model._model->network.output_size, _model._model->network.hidden_layers_size, _model._model->network.hidden_layers_count)) {}

double BackPropagation::get_cross_entropy_loss(const vector<double> &prediction, const int target) {
	return -log(prediction[target]);
}

double BackPropagation::get_total_error(const neural_network &temp_network, const int target) {
	return get_cross_entropy_loss(
	    temp_network.layers[temp_network.hidden_layers_count]->getOut(),
	    target);
}

double BackPropagation::clip_gradients(double value) {
	if (value > CLIP_GRADIENTS) {
		return CLIP_GRADIENTS;
	}

	if (value < -CLIP_GRADIENTS) {
		return -CLIP_GRADIENTS;
	}
	return value;
}

void BackPropagation::calculate_gradient(const Layer &layer, const vector<double> &deltas, const vector<double> &prevLayer, LayerParameters &gradient_) {
	calculate_gradient_for_bias(layer, deltas, gradient_);
	calculate_gradient_for_weights(layer, prevLayer, deltas, gradient_);
}

vector<double> BackPropagation::calculate_delta_for_hidden(const Layer &current_layer, const Layer &next_layer, const vector<double> &next_deltas) {
	vector<double> deltas(current_layer.getSize(), 0.0);

	for (int i = 0; i < current_layer.getSize(); i++) {
		deltas[i] = 0.0;
		for (int j = 0; j < next_layer.getSize(); j++) {
			deltas[i] += next_deltas[j] * next_layer.getWeight(j, i);
		}

		deltas[i] *= ActivationFunctions::DerivativeRelu(current_layer.getDots().net[i]);
	}

	return deltas;
}

void BackPropagation::calculate_gradient_for_bias(const Layer &layer, const vector<double> &deltas, LayerParameters &gradients) {
	for (int i = 0; i < layer.getSize(); i++) {
		gradients.bias[i] += clip_gradients(deltas[i]);
	}
}

void BackPropagation::calculate_gradient_for_weights(const Layer &layer, const vector<double> &prevLayer, const vector<double> &deltas, LayerParameters &gradients) {
	for (int i = 0; i < layer.getSize(); i++) {
		for (int j = 0; j < layer.getPrevSize(); j++) {
			gradients.weights[i][j] += clip_gradients(deltas[i] * prevLayer[j]);
		}
	}
}

vector<double> BackPropagation::calculate_delta_for_output(const vector<double> &out, const int target) {
	vector<double> deltas(out);

	deltas[target] -= 1.0;

	return deltas;
}

void BackPropagation::calculate_pattern_gradients(const TrainBoard &target, gradient &_gradients, const neural_network &temp_network) {
	vector<double> deltas;

	for (int layer_index = _gradients.gradients.size() - 1; layer_index >= 0; layer_index--) {
		Layer &layer = *temp_network.layers[layer_index];

		if (layer.getType() == OUTPUT) {
			deltas = calculate_delta_for_output(layer.getOut(), target.best_next_move);
		} else {
			deltas = calculate_delta_for_hidden(layer, *temp_network.layers[layer_index + 1], deltas);
		}

		if (layer_index == 0) {
			calculate_gradient(layer, deltas, target.board, _gradients.gradients[layer_index]);
		} else {
			calculate_gradient(layer, deltas, temp_network.layers[layer_index - 1]->getOut(), _gradients.gradients[layer_index]);
		}
	}
}

double BackPropagation::run_back_propagation(const TrainBoard &board, gradient &local_gradient) {
	model.run_model(board.board);
	calculate_pattern_gradients(board, local_gradient, model._model->network);

	return get_total_error(model._model->network, board.best_next_move);
}

double BackPropagation::run_back_propagation(const vector<TrainBoard> &boards, const double learning_rate) {
	const int batch_size = boards.size();
	const neural_network &network = model._model->network;
	double error = 0.0;

	global_gradient.reset();

	for (int i = 0; i < batch_size; i++) {
		error += run_back_propagation(boards[i], global_gradient);
	}

	update_weights(batch_size, global_gradient, learning_rate);
	return error / batch_size;
}

void BackPropagation::update_weights(int batch_size, gradient &gradients, double learning_rate) {
	gradients.multiply(-learning_rate / batch_size);
	model._model->updateWeights(gradients);
}
