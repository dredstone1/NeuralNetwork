#include "learning_rate.hpp"

namespace nn {
LearningRate::LearningRate(const DecayType decay_type_, const Global::ValueType initial_lr)
    : current_learningRate(initial_lr),
      decay_type(decay_type_) {
}
} // namespace nn
