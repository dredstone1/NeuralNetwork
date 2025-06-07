#ifndef LEARNINGRATE
#define LEARNINGRATE

#include "../model/config.hpp"

namespace nn {
class LearningRate {
  private:
	Global::ValueType current_learningRate;
	const DecayType decay_type;

  public:
	LearningRate(const DecayType decay_type_, const Global::ValueType initial_lr);
	void update_batch();
	Global::ValueType getLearningRate() { return current_learningRate; }
	~LearningRate() = default;
};
} // namespace nn
#endif // LEARNINGRATE
