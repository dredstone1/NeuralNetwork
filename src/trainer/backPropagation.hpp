#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "../model/Layers/Hidden_Layer.hpp"
#include "AiModel.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"

#define CLIP_GRADIENTS 0.001

class BackPropagation {
  private:
	AiModel &model;
	double get_total_error(const neural_network &temp_network, const int target);
	double get_cross_entropy_loss(const vector<double> &prediction, const int target);
	void calculate_pattern_gradients(const TrainSample &targer, gradient &gradients, const neural_network &temp_network);
	void update_weights(int bash_size, gradient &gradients, double learning_rate);
	void calculate_gradient(const Layer &layer, const vector<double> &deltas, const vector<double> &prevLayer, LayerParameters &gradients);
	double run_back_propagation(const TrainSample &sample, gradient &local_gradient);
	vector<double> calculate_delta_for_hidden(const Hidden_Layer &current_layer, const Layer &next_layer, const vector<double> &next_deltas);
	vector<double> calculate_delta_for_output(const vector<double> &out, const int target);
	void calculate_gradient_for_weights(const Layer &layer, const vector<double> &prevLayer, const vector<double> &deltas, LayerParameters &gradients);

  public:
	BackPropagation(AiModel &_model);
	double run_back_propagation(const Batch &batch, const double learning_rate);
	~BackPropagation() = default;
};
#endif // BACKPROPAGATION
