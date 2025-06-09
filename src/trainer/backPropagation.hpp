#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "../model/Layers/Hidden_Layer.hpp"
#include "Globals.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"
#include "learning_rate.hpp"

namespace nn::training {
constexpr global::ValueType MIN_ABS_CLIPPING_VALUE = 0.0001;
constexpr global::ValueType MAX_ABS_CLIPPING_VALUE = 10;

constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

class BackPropagation {
  private:
	gradient local_gradient;
	AiModel &model;
	LearningRate &learningRate;
	global::ValueType get_total_error(const model::NeuralNetwork &temp_network, const int target);
	global::ValueType get_cross_entropy_loss(const global::ParamMetrix &prediction, const int target);
	void calculate_pattern_gradients(const TrainSample &targer, const model::NeuralNetwork &temp_network);
	void update_weights(int bash_size, global::ValueType learning_rate);
	void calculate_gradient(const model::Layer &layer, const global::ParamMetrix &deltas, const global::ParamMetrix &prevLayer, model::LayerParameters &gradients);
	global::ValueType run_back_propagation(const TrainSample &sample);
	global::ParamMetrix calculate_delta_for_hidden(const model::Hidden_Layer &current_layer, const model::Layer &next_layer, const global::ParamMetrix &next_deltas);
	global::ParamMetrix calculate_delta_for_output(const global::ParamMetrix &out, const int target);
	void calculate_gradient_for_weights(const model::Layer &layer, const global::ParamMetrix &prevLayer, const global::ParamMetrix &deltas, model::LayerParameters &gradients);
	static global::ValueType clippValue(const global::ValueType value);

  public:
	BackPropagation(AiModel &_model, LearningRate &learningRate_);
	global::ValueType run_back_propagation(const Batch &batch);
	~BackPropagation() = default;
};
} // namespace nn::training
#endif // BACKPROPAGATION
