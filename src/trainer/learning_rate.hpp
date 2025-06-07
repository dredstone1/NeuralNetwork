#ifndef LEARNINGRATE
#define LEARNINGRATE

#include "../model/config.hpp"
#include "Globals.hpp"

namespace nn {
class LearningRate {
  private:
	Global::ValueType current_learningRate;
	const TrainingConfig &config;

  public:
	LearningRate(const TrainingConfig &config);
	void update_batch(const int current_batch, const Global::ValueType error);
	Global::ValueType getLearningRate() { return current_learningRate; }
	~LearningRate() = default;
};
} // namespace nn
#endif // LEARNINGRATE
