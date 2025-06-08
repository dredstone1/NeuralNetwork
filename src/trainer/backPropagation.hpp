#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "../model/Layers/Hidden_Layer.hpp"
#include "Globals.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"
#include "learning_rate.hpp"
#include <vector>

namespace nn {
constexpr Global::ValueType MIN_ABS_CLIPPING_VALUE = 0.0001;
constexpr Global::ValueType MAX_ABS_CLIPPING_VALUE = 10;

constexpr Global::ValueType MIN_LOSS_VALUE = 1e-10;

class BackPropagation {
  private:
	gradient local_gradient;
	AiModel &model;
	LearningRate &lr;
	Global::ValueType get_total_error(const neural_network &temp_network, const int target);
	Global::ValueType get_cross_entropy_loss(const std::vector<Global::ValueType> &prediction, const int target);
	void calculate_pattern_gradients(const TrainSample &targer, const neural_network &temp_network);
	void update_weights(int bash_size, Global::ValueType learning_rate);
	void calculate_gradient(const Layer &layer, const std::vector<Global::ValueType> &deltas, const std::vector<Global::ValueType> &prevLayer, LayerParameters &gradients);
	Global::ValueType run_back_propagation(const TrainSample &sample);
	std::vector<Global::ValueType> calculate_delta_for_hidden(const Hidden_Layer &current_layer, const Layer &next_layer, const std::vector<Global::ValueType> &next_deltas);
	std::vector<Global::ValueType> calculate_delta_for_output(const std::vector<Global::ValueType> &out, const int target);
	void calculate_gradient_for_weights(const Layer &layer, const std::vector<Global::ValueType> &prevLayer, const std::vector<Global::ValueType> &deltas, LayerParameters &gradients);
	Global::ValueType clippValue(const Global::ValueType value);
	void addNoise(std::vector<Global::ValueType> &input, Global::ValueType noise_level = 0.05);

  public:
	BackPropagation(AiModel &_model, LearningRate &lr_);
	Global::ValueType run_back_propagation(const Batch &batch);
	~BackPropagation() = default;
};
} // namespace nn
#endif // BACKPROPAGATION
