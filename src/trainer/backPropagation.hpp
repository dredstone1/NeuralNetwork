#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "../model/Layers/Hidden_Layer.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"

namespace nn {
class BackPropagation {
  private:
	gradient local_gradient;
	AiModel &model;
	Global::ValueType get_total_error(const neural_network &temp_network, const int target);
	Global::ValueType get_cross_entropy_loss(const std::vector<Global::ValueType> &prediction, const int target);
	void calculate_pattern_gradients(const TrainSample &targer, const neural_network &temp_network);
	void update_weights(int bash_size, Global::ValueType learning_rate);
	void calculate_gradient(const Layer &layer, const std::vector<Global::ValueType> &deltas, const std::vector<Global::ValueType> &prevLayer, LayerParameters &gradients);
	Global::ValueType run_back_propagation(const TrainSample &sample);
	std::vector<Global::ValueType> calculate_delta_for_hidden(const Hidden_Layer &current_layer, const Layer &next_layer, const std::vector<Global::ValueType> &next_deltas);
	std::vector<Global::ValueType> calculate_delta_for_output(const std::vector<Global::ValueType> &out, const int target);
	void calculate_gradient_for_weights(const Layer &layer, const std::vector<Global::ValueType> &prevLayer, const std::vector<Global::ValueType> &deltas, LayerParameters &gradients);

  public:
	BackPropagation(AiModel &_model);
	Global::ValueType run_back_propagation(const Batch &batch, const Global::ValueType learning_rate);
	~BackPropagation() = default;
};
} // namespace nn
#endif // BACKPROPAGATION
