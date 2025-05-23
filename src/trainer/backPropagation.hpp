#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "AiModel.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"
#include "dataBase.hpp"

#define CLIP_GRADIENTS 0.001

class BackPropagation {
  private:
	AiModel &model;
	gradient global_gradient;
	static double get_total_error(const neural_network &temp_network, const int target);
	static double get_cross_entropy_loss(const vector<double> &prediction, const int target);
	static void calculate_pattern_gradients(const TrainSample &targer, gradient &gradients, const neural_network &temp_network);
	void update_weights(int bash_size, gradient &gradients, double learning_rate);
	static void calculate_gradient(const Layer &layer, const vector<double> &deltas, const vector<double> &prevLayer, LayerParameters &gradients);
	static double clip_gradients(double value);
	double run_back_propagation(const TrainSample &sample, gradient &local_gradient);
	static vector<double> calculate_delta_for_hidden(const Layer &current_layer, const Layer &next_layer, const vector<double> &next_deltas);
	static vector<double> calculate_delta_for_output(const vector<double> &out, const int target);
	static void calculate_gradient_for_bias(const Layer &layer, const vector<double> &deltas, LayerParameters &gradients);
	static void calculate_gradient_for_weights(const Layer &layer, const vector<double> &prevLayer, const vector<double> &deltas, LayerParameters &gradients);

  public:
	BackPropagation(AiModel &_model);
	double run_back_propagation(const Batch &batch, const double learning_rate);
	~BackPropagation() = default;
};
#endif // BACKPROPAGATION
