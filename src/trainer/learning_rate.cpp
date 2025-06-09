#include "learning_rate.hpp"

namespace nn::training {
LearningRate::LearningRate(const model::TrainingConfig &config_)
    : current_learningRate(config_.lr_config.lr_init_value),
      config(config_) {
}
} // namespace nn::training
