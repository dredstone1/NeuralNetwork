#include "learning_rate.hpp"
#include "Globals.hpp"
#include "model/config.hpp"

namespace nn {
LearningRate::LearningRate(const TrainingConfig &config_)
    : current_learningRate(config_.lr_config.lr_init_value),
      config(config_) {
}

void LearningRate::update_batch(const int current_batch, const Global::ValueType error) {
	switch (config.lr_config.decay_type) {
	case DecayType::Constant:
		break;

	case DecayType::StepDecay: {
		break;
	}

	case DecayType::ExponentialDecay: {
		break;
	}

	default:
		break;
	}
}
} // namespace nn
