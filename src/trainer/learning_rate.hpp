#ifndef LEARNINGRATE
#define LEARNINGRATE

#include "../model/config.hpp"

namespace nn::training {
struct LearningRate {
	global::ValueType current_learningRate;
	const model::TrainingConfig &config;
	LearningRate(const model::TrainingConfig &config);
	global::ValueType getLearningRate() { return current_learningRate; }
	~LearningRate() = default;
};
} // namespace nn::training
#endif // LEARNINGRATE
