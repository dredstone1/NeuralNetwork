#ifndef BACKPROPAGATION
#define BACKPROPAGATION

#include "Globals.hpp"
#include "dataBase.hpp"
#include "gradient.hpp"

namespace nn::training {
constexpr global::ValueType MIN_LOSS_VALUE = 1e-10;

struct learningRateParams {
	global::ValueType currentLearningRate;
	learningRateParams(const global::ValueType initLearningRate) : currentLearningRate(initLearningRate) {}
};

class BackPropagation {
  private:
	gradient local_gradient;
	AiModel &model;
	learningRateParams &learningRate;

	static global::ValueType L1Regularization(const global::ValueType weight);
	static global::ValueType get_total_error(const model::NeuralNetwork &temp_network, const int target);
	static global::ValueType get_cross_entropy_loss(const global::ParamMetrix &prediction, const int target);
	void calculate_pattern_gradients(const TrainSample &targer, const model::NeuralNetwork &temp_network);
	void update_weights(int bash_size);
	void calculate_gradient(const model::Layer &layer, const global::ParamMetrix &deltas, const global::ParamMetrix &prevLayer, model::LayerParameters &gradients);
	void calculateGradientForWeights(const model::Layer &layer, const global::ParamMetrix &prevLayer, const global::ParamMetrix &deltas, model::LayerParameters &gradients);
	void calculateGradientForBiases(const model::Layer &layer, const global::ParamMetrix &deltas, model::LayerParameters &gradients);

	global::ValueType run_back_propagation(const TrainSample &sample);
	global::ParamMetrix calculateDeltaHidden(const model::Hidden_Layer &current_layer, const model::Layer &next_layer, const global::ParamMetrix &next_deltas);
	global::ParamMetrix calculateDeltaForOutput(const global::ParamMetrix &out, const int target);

  public:
	BackPropagation(AiModel &_model, learningRateParams &learningRate_);
	~BackPropagation() = default;

	global::ValueType run_back_propagation(const Batch &batch);
};
} // namespace nn::training
#endif // BACKPROPAGATION
